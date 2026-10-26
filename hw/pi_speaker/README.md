# Speaker-in-a-room experiment (Raspberry Pi 5 + PAM8302 amp)

## Why

The detector is trained only on simulated audio. The least tested part of the simulator is the
"muffler" in `model/mix.py` (`Mixer.placement()`), which models what happens when somebody
talks from 0.3–5 m away:

- level drops 6 dB per doubling of distance;
- reverberation grows with distance;
- the sound gets duller.

This experiment measures what a real room actually does to the same clips, using a speaker
as the talker. It compares the result with the simulator in the chip's own feature domain
(the 16 log bands). It reports whether the simulator is realistic and which knob to turn if
it isn't.

> The amp (PAM8302: A+, A−, SD, Vin 2–5 VDC, GND) plays sound; it doesn't record. The
> recording side is normally the two PDM mics and the always-on recorder in `../pi_mics/`
> (`--rec pdm`). A USB mic or a phone also works.

## What you need

| item | note |
|---|---|
| Raspberry Pi 5 + PAM8302 board | |
| Speaker, 4–8 Ω, ≥ 2 W | Anything from a 40 mm full-range driver to a salvaged laptop speaker. |
| **USB audio adapter** with headphone out | Needed because the **Pi 5 has no analog audio jack**. A $5–10 "USB sound card" works. One with a mic input can also do the recording. |
| A recorder | The PDM mics with `../pi_mics/record.py` running (`--rec pdm`), or a USB mic or phone. |
| 3.5 mm cable or plug, jumper wires | To bring the adapter's output to A+/A−. |
| Phone with an SPL-meter app | Optional. Lets you set a realistic speaking level. |

## Wiring

```
 Raspberry Pi 5 header                      PAM8302 board
 ─────────────────────                      ─────────────
 pin 2  (5V)  ───────────────────────────── Vin (2-5 VDC)
 pin 6  (GND) ───────────────────────────── GND
 pin 11 (GPIO17) ────────────────────────── SD   (high = on, low = shutdown)

 USB audio adapter, headphone plug          PAM8302 board
 ─────────────────────────────────          ─────────────
 Tip    (left channel) ──────────────────── A+
 Sleeve (ground)       ──────────────────── A−
 Ring   (right channel)   not connected

 PAM8302 output (+ / −) ─────────────────── speaker (two wires, either way round)
```

- **A+/A− is a differential input.** For a single-ended source like a headphone out, signal
  goes to A+ and ground to A−. That is the standard hookup for this board.
- **Never connect either speaker output terminal to ground,** or to a scope or recorder
  ground. The output is bridged (BTL), and grounding it can kill the amp.
- **SD is optional.** The board pulls SD high (always on) if you leave it unconnected.
  Wiring it to GPIO17 lets the script mute the amp between takes, so its hiss doesn't end
  up in your silences.
- **Power:** the Pi's 5 V pin can supply the amp, which draws at most ~0.6 A at full volume
  into 4 Ω. Use the official 5 V/5 A supply.
- **Volume:** the small trim pot on the board sets the volume. Set it once with `level`
  (below), then **don't touch it again** for the rest of the session.

## Software

1. On the workstation, in the repo env (`. ./env.sh`), build the playback file (129 s). It
   contains:
   - sync sweeps;
   - a 1 kHz tone and pink noise;
   - 30 held-out TTS "telescope" clips, 10 near-misses and 10 conversation utterances, all
     at equal RMS.

   ```
   python hw/pi_speaker/make_stimuli.py              # --word <slug> for another candidate
   scp -r hw/pi_speaker pi@<pi>:~/
   ```

2. On the Pi, nothing to install: Raspberry Pi OS ships python3, gpiozero, aplay and
   arecord.

   ```
   cd ~/pi_speaker
   python3 play_record.py devices                        # find the card numbers
   python3 play_record.py level --play plughw:2,0        # loops pink noise; set the pot
   python3 play_record.py run --play plughw:2,0 --rec plughw:2,0 --dist 0.1 --note "ref, living room"
   ```

   - In `devices`, the USB adapter shows up as e.g. "card 2: Device". Use `plughw:<card>,0`
     for both playback and recording if it has a mic input.
   - During `level`, aim for **~65–70 dB(A) at 0.5 m** on the SPL app. That is normal
     conversational speech; the simulator's "talker at 0.5 m" is meant to be the same
     thing. Pass the reading with `--spl 68`.

3. Copy `recordings/` back to the workstation and analyse. Use any checkpoint for
   `--model`; it adds detector scores on real vs simulated keyword clips.

   ```
   scp -r pi@<pi>:~/pi_speaker/recordings hw/pi_speaker/
   python hw/pi_speaker/analyze.py --model data/results/tune3_h32/telescope.pt
   ```

   This writes `hw/pi_speaker/recordings/report.md`.

## Protocol (about 30 min)

1. **Room and placement**
   - Use an ordinary room: living room or kitchen, not a closet full of pillows.
   - Put the **recorder (the "chip") on a table**, where a phone or device would sit.
   - Put the **speaker at mouth height (~1.5 m), pointing at the recorder**.
   - Keep the room quiet: no fans or TV for the main series.
2. **Reference at 0.1 m.** Record with the speaker 10 cm from the recorder. This one
   recording captures the speaker's and recorder's own coloration, and the analysis
   divides it out. Every later take is compared against it.
3. **Distance series,** measured speaker-to-recorder: **0.5, 1, 2, 3, 4.5 m**, one `run`
   each with `--dist`.
4. **Repeat** one distance (e.g. 2 m) at a slightly different spot. This shows how much
   two "identical" real takes differ, which is the noise floor for judging the simulator.
5. **Muffled** takes (optional, use `--note`):
   - recorder in a jacket pocket, speaker at 1 m;
   - speaker in the next room with the door open.

   These test the simulator's 15% "muffled" case. The table compares them with the
   unmuffled model, so expect big shape differences there; that is the measurement.
6. **Second room** (optional): repeat steps 2–4 in a room that sounds different (tiled
   kitchen or bathroom vs carpeted bedroom).

**Naming:** `--dist` is all the analysis needs. Each take writes `<label>.wav` and
`<label>.json` (distance, note, SPL, devices).

## Recording with a phone instead

Run `play_record.py run` without `--rec`, and record the whole 129 s on the phone. Then:

1. Save the recording as WAV (export or convert it, e.g.
   `ffmpeg -i x.m4a -ac 1 -ar 16000 x.wav`).
2. Put it in `recordings/` under the printed label.
3. Keep the `.json` that `run` wrote next to it.

The sync sweeps at the start and end let `analyze.py` find every clip and correct the
phone's clock drift. Use a recorder app with **automatic gain control and noise
suppression turned off** (many voice-memo apps have a "raw" or "unprocessed" mode). AGC
undoes exactly the level drop we're trying to measure.

## What the report tells us

Each row compares one recording at distance d with the reference passed through the
simulator at the same d (20 random simulated rooms per clip):

| column | real vs sim | if it's off, change in `mix.placement()` |
|---|---|---|
| level drop | e.g. −15 dB real vs −20 dB sim | Real rooms hold level up better than 1/d (reverberant field), so the gain law needs a reverberant term. |
| shape diff vs sim spread | shape error should be ≤ the spread | The per-band table shows the direction. Highs too weak in real → the distance low-pass is too gentle; highs too strong → too harsh. |
| tail | real inside sim p10..p90 | Real above the range: the sim is too dry (lower the critical distance). Real below: the sim is too wet. |
| keyword score | similar medians | If real scores clearly lower, the detector relies on something the simulator gets wrong. |

These numbers say how far to trust the simulator at each distance. We then tune
`placement()` and re-check against these same recordings. They become a permanent
regression test for the simulator.

The analysis pipeline was checked on fake recordings: stimuli through real recorded room
impulse responses at 0.1/1/3 m, resampled to 48 kHz with 80 ppm clock drift. It recovers
the clock error and produces the full table.

## With the PDM mics

1. Run `../pi_mics/record.py` (or its service).
2. Use `--rec pdm --dist <mic0 m> --dist1 <mic1 m>`. Each take is cut out of the continuous
   recording for both mics.
3. `analyze.py` gives a separate table per mic, each against its own 0.1 m reference.

The mics' audio is decimated bit-exact to the chip's CIC, so this tests the simulator
against what the chip itself would hear.
