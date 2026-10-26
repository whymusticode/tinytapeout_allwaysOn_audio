# Always-on two-mic recorder (Raspberry Pi 5 + 2 PDM mics)

## Why

Two PDM MEMS mics record continuously through **exactly the chip's front end**:

- the same 1.024 MHz mic clock;
- the same 3rd-order CIC decimator, bit-exact to `src/pdm_cic.v`;
- 16 kHz int16 output.

What comes out of the Pi is what the chip's filterbank would see from the same mic. That
gives us:

1. **Real false-alarm audio** from the place the device will live: hours of household
   conversation, TV, kitchen and street noise. This becomes a real test stream next to the
   AMI meetings, and it is the most realistic false-alarm test we can get.
2. **Real labelled keywords** (`label.py`): you and others saying candidate words and their
   near-misses at known distances. These become real positives for the test set.
3. **The real mic's own response and noise floor** in the simulator's terms, to check the
   level, spectrum and self-noise assumptions in `model/mix.py`.
4. **Raw PDM bits** (`--raw`): mic data that can later be replayed through the RTL or the
   FPGA, bit for bit.

All of this is **test data**. Training stays 100% simulated (see `model/README.md`). The
better the simulator predicts these recordings, the more we can trust it.

## The mics

The 3V / GND / SEL / CLK / DAT pins mean a PDM MEMS breakout (for example Adafruit #3492,
MP34DT01-M):

- supply 1.6–3.6 V;
- clock 1.0–3.25 MHz; we run at 1.024 MHz, the chip's PDM clock;
- SEL picks which half of the clock the mic drives DAT in: GND = one half, 3V = the other.

## Wiring

The clock is shared and each mic gets its own data wire, so the mics don't have to sit
next to each other.

| Pi 5 header pin | signal | mic 0 | mic 1 |
|---|---|---|---|
| 1 (3V3) | power | 3V | 3V |
| 9 (GND) | ground | GND | GND |
| 29 (GPIO5) | CLK out, 1.024 MHz | CLK | CLK |
| 31 (GPIO6) | DAT0 in | DAT | — |
| 26 (GPIO7) | DAT1 in | — | DAT |
| — | channel select | SEL → GND | SEL → GND |

```
  Pi 5 header (pin 1 top-left, USB ports at the bottom)

   3V3  (1) ●──────────┬──────────── mic0 3V
                       └──────────── mic1 3V
   GND  (9) ●──────────┬──────────── mic0 GND, mic0 SEL
                       └──────────── mic1 GND, mic1 SEL
 GPIO5 (29) ●──────────┬──────────── mic0 CLK
                       └──────────── mic1 CLK
 GPIO6 (31) ●─────────────────────── mic0 DAT
 GPIO7 (26) ●─────────────────────── mic1 DAT
```

- **Either SEL setting works.** The capture program samples both data pins on both clock
  halves, and the recorder works out which half each mic uses. It prints the matching
  `PDM_EDGE` value for the chip's `ui[1]` pin.
- **Shared-wire stereo also works.** That is the classic arrangement: both DATs on one
  wire, one SEL to GND and the other to 3V. With that wiring both mics come out of DAT0's
  two halves; the separate-wire layout above is just simpler.
- **Keep wires short.**
  - Up to ~30 cm with ordinary jumpers is fine.
  - Up to ~1 m works if you run a GND wire alongside CLK and each DAT (a twisted pair is
    best).
  - Beyond that, move the talker rather than the mic.
- **GPIO7 is SPI0's CE1.** Leave SPI disabled (the default) or use different pins: change
  `PIN_DAT0` in `pdm_capture.c`. DAT1 must be PIN_DAT0 + 1.
- **The speaker amp, if you use it,** needs pins 2, 6 and 11 (see `../pi_speaker/`). None of
  them collide with these.

## Setup (on the Pi)

1. Copy this folder to the Pi and run the setup script:

   ```
   scp -r hw/pi_mics pi@<pi>:~/
   ssh pi@<pi>
   cd ~/pi_mics && ./setup.sh
   ```

   The script:
   - needs `/dev/pio0` (a current Raspberry Pi OS; `sudo apt full-upgrade` if it's missing);
   - builds Raspberry Pi's `piolib`, then `pdm_capture`;
   - makes `.venv` with numpy and soundfile.

2. Check the mics:

   ```
   sudo .venv/bin/python record.py --out ~/recordings
   ```

   The recorder logs each mic's level every 10 s. To check it's working:
   - **Both mics found:** you should see `mic0: found on the low clock half (chip PDM_EDGE=1)`
     and the same for mic1.
   - **Which mic is which:** tap one, and its level jumps.
   - **Levels:** a quiet room reads somewhere around −60 to −70 dBFS, and normal speech at
     1 m roughly −35 to −45 dBFS.
   - **`behind` should stay near 0 ms.** If it grows, the Pi is dropping audio; tell me and
     I'll make the reader faster.

3. Run it always-on as a service:

   ```
   ./setup.sh service                # restarts on boot and on crashes
   journalctl -u pdm-record -f      # the level log
   ```

## Experiments

### E0: are the two mics the same? (10 min, do this first)

Put both mics **side by side** on the table and talk or play anything for a few minutes.
They should agree within ~1 dB in every band. If they do, any later difference between
them comes from placement, not the parts. I'll compare them in the notebook from the
recordings.

### E1: always-on household recording (runs indefinitely)

Leave the recorder running as a service.

**Placement:**
- **mic 0 = "the device":** open, on a table or shelf where a gadget would sit. It stays
  there.
- **mic 1 = the variation,** changed every few days and noted:
  1. inside a jacket pocket or a fabric pouch, which tests the simulator's "muffled" case;
  2. inside a small box with a hole, standing in for a product enclosure;
  3. side by side again, as a periodic sanity check.

**Storage:** about **3 GB/day** for both mics as FLAC. `--raw` adds about 22 GB/day, so
keep raw only for short sessions.

**Privacy:** this records everything said in the room. Tell the people who live or visit
there. The data stays on the Pi and your workstation; `recordings/` is git-ignored and
never goes into the repo. If anyone asks, drop a day: delete that day's folder.

**What it gives:** a real false-alarm stream for the exact environment. "How many false
alarms per hour does the chip produce in your kitchen?" becomes a measured number. It also
checks whether the simulator's conversation and ambience mix matches real life.

### E2: labelled words (sessions of ~5 min)

While the recorder runs, from any terminal on the Pi (ssh is fine):

```
python3 label.py --speaker mike --d0 1.0 --d1 1.0 --n 40 --note "living room, quiet"
```

1. The script shows a word (a candidate, or about 30% of the time a near-miss of one).
2. Say it once, naturally. It moves on after 2.5 s, or press Enter to go on early.
3. `--d0` / `--d1` are your distances to each mic in metres.

**Vary across sessions:**
- distance: 0.5, 1, 2, 3+ m;
- facing toward or away from the mics;
- a quiet room vs TV, dishwasher or music on;
- **other speakers:** different voices matter most for the real FRR, and each session uses
  a new `--speaker` id;
- `--only telescope,camel` to concentrate on the words that are doing well.

Labels go to `labels/<date>_<speaker>.jsonl` with wall-clock times. The audio is cut from
the continuous recordings on the workstation.

### E3: speaker in a room (optional, needs the amp + a USB sound card)

`../pi_speaker/` plays a fixed set of held-out TTS clips through a speaker at 0.1–4.5 m and
compares the room's real effect with the simulator's "muffler". With the PDM mics the
recording side is automatic:

```
python3 play_record.py run --play plughw:2,0 --rec pdm --dist 2.0 --dist1 2.0
```

Use `sudo` if the recorder runs as a service.

### E4: chip in the loop (when the FPGA works)

`record.py --raw` keeps the mic's actual 1.024 MHz bits. Replaying them through the RTL
(cocotb) or into the FPGA's `ui[0]` lets us compare the chip's features and detections
with the Python model on identical real input.

## Getting the data to the workstation

```
rsync -av pi@<pi>:recordings/ data/raw/pi_mics/recordings/
rsync -av pi@<pi>:pi_mics/labels/ data/raw/pi_mics/labels/
```

`data/` is git-ignored. Next on my side, once data arrives:
- a `build_stream.py` stream `real_home` from E1;
- real positives from E2 cut out by label times;
- the E0 mic-match check.

## Files

| file | runs on | what |
|---|---|---|
| `pdm_capture.c` | Pi | PIO program: 1.024 MHz clock on GPIO5, samples GPIO6/7 on both clock halves, streams to stdout |
| `pdm.py` | Pi + workstation | unpack the capture, find each mic's clock half, CIC decimator bit-exact to `pdm_cic.v` |
| `record.py` | Pi | always-on recorder: rotating 10-min FLAC per mic + JSON (start time, PDM edge); `--raw` keeps PDM bits; `extract()` cuts by wall-clock time |
| `label.py` | Pi | prompts words, logs wall-clock windows for labelled positives / near-misses |
| `words.txt` | — | prompts: 66 candidates + 2 near-misses each (from `model/wakewords.yaml`) |
| `setup.sh`, `pdm-record.service`, `requirements.txt` | Pi | build, venv, systemd service |
| `test_pdm.py` | workstation | `python -m pytest hw/pi_mics/test_pdm.py`: decoder == `model/frontend.py:cic3()` bit for bit |

## Status

- **Verified here, without hardware:**
  - the decoder is bit-exact to the chip's CIC golden model, fed in uneven chunks;
  - phase detection works;
  - the recorder runs end to end on a simulated capture of two sigma-delta mics: FLAC and
    raw bits both exact.
- **Not yet run on a real Pi 5:** `pdm_capture.c` and `setup.sh`. The likely snags:
  - piolib's build or library names (the script searches for the library);
  - `/dev/pio0` permissions (run the recorder with `sudo`);
  - PIO clock rate (the program prints it at start).

  Send me the output if anything fails.
