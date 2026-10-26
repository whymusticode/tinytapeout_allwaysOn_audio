# Model: wake-word selection and sizing

Purpose: find which wake word is most reliably detectable, and how small a streaming
model can be while still working reliably. The results set the chip's tile count and memory needs.

## Three models of the same chip

| Tier | Where | Use | Checked by |
|---|---|---|---|
| Golden (bit-exact, scalar, readable) | `frontend.py` | Reference for the RTL. Slow (~100 ms per 1.6 s of audio). | `test/` and `test/unit/` (cocotb) |
| Fast (bit-exact, numba, parallel) | `features.py` | Generates all training and evaluation features (~1 ms per 1.6 s window per core). | `test_bitexact.py` (pytest) |
| Verilog | `src/*.v` | The chip. Constants are generated from the golden model by `gen_rtl.py`. | |

The front end is bit-exact from int16 PCM onward. The step from float audio to PCM
(`features.pcm_from_float`) models the mic and CIC droop, and the CIC itself is checked
bit-exact from PDM bits in `test/test.py`. The classifier will get the same three tiers
(torch with quantization-aware training, an integer numpy golden model, and RTL) once the sweep has fixed its size.

```
python -m pytest model/test_bitexact.py -q     # fast == golden
cd test/unit && make                           # RTL == golden (per module)
cd test && make                                # full chip (PDM in)
python model/gen_rtl.py                        # regenerate src/*_coeffs.vh after changing the golden model
```

## Memory rules

The previous `build_cache` run used >100 GB and froze the machine, because each of the 30 workers
`np.load`ed its own private copy of every `.npz` pool. Now:

- **Datasets are stored as directories of `.npy` files** (`store.py`) and memory-mapped, so workers
  share one copy in the page cache.
- **Builders write into a preallocated memory-mapped output** instead of collecting lists.
- **Run heavy jobs through `model/capped.sh`** (`MEM=30G model/capped.sh python model/build_cache.py`).
  It runs the job in a systemd scope with a hard memory cap, so it gets killed instead of freezing the machine, and prints the
  job's peak memory.

Measured peaks: `build_pools` < 40 GB cap, `build_cache` train 1.4 GB, `build_stream` 3.7 GB,
one `train.py` 0.5 GB.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Download open datasets | `data/download.sh` | `data/raw/` |
| 2. Synthesize candidates (Piper TTS, 4 voices / ~1050 speakers) | `synth_tts.py`, `synth_tts.py --negatives` | `data/synth/`, `data/synth_neg/` |
| 3. Background / RIR / speech pools, train/test split | `build_pools.py` | `data/pools/` |
| 4. Mixed feature windows for all candidates | `build_cache.py` | `data/cache/{train,test}/` |
| 5. Held-out false-alarm stream (~15 h) | `build_stream.py` | `data/cache/stream/` |
| 6. Train + evaluate every candidate | `sweep.py` | `data/results/<run>/REPORT.md` |

Run everything from the repo root after `. ./env.sh`.

## Candidates

`wakewords.yaml` has 105 candidates: Linux-themed phrases, "hey/okay X", long uncommon words,
invented words and longer phrases. It also has two control groups:

- **`control_real`:** Speech Commands words. Their models are trained on synthetic speech only, then
  tested on real recordings, which measures how well synthetic training transfers to real voices.
- **`control_bad`:** common words that should rank poorly, as a sanity check on the ranking.

## Data hygiene

- **Test speakers are held out.** TTS speakers are split by id, so every 5th speaker of each voice is test-only.
- **Backgrounds, rooms and the false-alarm stream are held out.**
  - Test backgrounds: ESC-50 fold 5, a hashed MUSAN split, and DEMAND environments PCAFETER, STRAFFIC, TMETRO, DLIVING and NPARK.
  - Test rooms: real recorded RIRs are test-only. Training uses simulated RIRs.
  - False-alarm stream: LibriSpeech is used only in the stream.
- **Real recordings of a word are never used to train that word's own model.**

## Features

Integer, exactly as the chip computes them (`frontend.filterbank`). 16 mel-spaced band-pass biquads
(150 Hz to 7 kHz, Q2.14 coefficients, 4 guard bits in the 22-bit state) feed a sum of |y| over each 16 ms frame
(24-bit, saturating), which is turned into a log2 code in 1/8-octave steps (uint8, 0..191).
The fixed-point widths were chosen by measurement against a float filterbank: the error is under
0.01 octave for normal frames and under 0.13 for the quietest, both below the 1/8-octave output step.

## Detector

`train.py` builds a GRU(H) followed by a linear layer, giving one score per 16 ms frame. Only the H-value state persists
between frames, which suits a chip with little memory. Positives must fire within 224 ms of the
word ending; negatives must never fire after the 0.5 s warm-up.
