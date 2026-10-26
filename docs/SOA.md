# State of the art: how the big players and chip designers build always-on wake words

What is publicly known, with references, and what each finding means for this chip.

Evidence labels:
- **(stated):** the source says it directly.
- **(derived):** computed by us from stated numbers; the arithmetic is shown.
- **(secondary):** a reputable summary of a primary source we couldn't read directly.

Last updated 2026-10-26.

## 1. The common pattern: a cascade

Apple, Google and Amazon all use a **cascade**:

1. A tiny, always-on first stage on a low-power processor. It is tuned to rarely miss and
   tolerates false alarms.
2. A much larger second stage on the main processor re-checks each trigger on buffered audio.
3. Often a third check: speaker verification, a cloud model, or a full-utterance
   false-trigger model.

| | first stage (always on) | where | size | second stage |
|---|---|---|---|---|
| **Apple** "Hey Siri" (2017) | DNN, 5 hidden layers × 32 units, sigmoid, over 20 stacked filter-bank frames, ~20 phonetic-state outputs + HMM-style scoring [A1] (stated) | Always On Processor (motion coprocessor) [A1] | **~13k–31k weights** (derived, §2.1) | same structure, 5 × 192 (~0.2–0.3M weights, derived) on the main CPU [A1] |
| **Apple** voice trigger (2023) | "low-power, high-recall" DNN-HMM, 23 states, end-to-end trained through a differentiable HMM decoder, **4-bit palettised weights**, input from a ring buffer [A2] (stated) | Always On Processor [A2] | not stated | conformer checker → LSTM speaker ID → three false-trigger-mitigation models [A2] |
| **Google** "Ok Google" (2017) | keyword model, log-mel 32–40 channels [G1] (stated) | power-isolated **DSP** [G1] | **13 kB footprint** [G1] (stated) | larger model on the application processor + LSTM speaker verification [G1] |
| **Google** E2E (2019) | 7 rank-1 **SVDF** layers + bottlenecks, 40 log-mel bands, one inference per 20 ms [G2] (stated) | phone | 40k / 318k / 700k params [G2] (stated) | (2-stage variant evaluated, see §3) |
| **Amazon** Alexa (2018) | DNN-HMM keyword decoder [Z1] (stated) | device | not stated | small feed-forward NN on device [Z1], then **cloud verification** [Z2] |

The always-on stage everywhere is roughly **10–40k weights**, and up to ~0.5M on dedicated
silicon (Syntiant, §4). This chip's 16-tile middle (29k–98k) is in that range.

## 2. Apple "Hey Siri"

### 2.1 The 2017 first stage [A1]

**Front end (stated)**
- 16 kHz audio, filter-bank frames every ~10 ms.
- "About twenty of these frames at a time (0.2 sec of audio) are fed to the acoustic model."

**Network (stated)**
- 5 hidden layers, all the same size: 32, 128 or 192 units depending on the device.
- Sigmoid nonlinearities.
- Output: about 20 classes, i.e. the phrase's phonetic states (3 per phoneme) plus silence and
  "other speech", softmax as log probabilities.

**First vs second pass (stated, via [A1s])**
- "the first detector used a DNN with 5 layers of 32 hidden units"
- "the second detector had 5 layers of 192 hidden units"
- Apple Watch runs a single medium-size model, after a wrist raise, in ~5% of its compute.

**Scoring (stated):** an HMM-style dynamic programme over the phrase states. For each state it
takes the max of staying or advancing, then adds the frame's log score. Stay/advance costs
come from duration statistics. The final score is the last state's.

**Thresholds (stated):** a normal threshold plus a lower "second-chance" threshold. A score
between them makes the system more sensitive for a few seconds, so a user who repeats the
phrase gets through.

**Targets and data (stated)**
- False alarms "as low as one per week"; negatives are "thousands of hours".
- Training uses real "Hey Siri" utterances plus general speech, with automatic transcription.

**Size (derived).** Apple doesn't give the number of filter-bank channels F, so we compute
weights + biases for two values:

```
(20·F·32 + 32) + 4·(32·32 + 32) + (32·C + C),   C = 20 (or 23 in the 2023 article)
F = 13: 13,236 (C = 20) / 13,335 (C = 23)
F = 40: 30,516 (C = 20) / 30,615 (C = 23)
second pass, 5 × 192, F = 13..40: 0.20M..0.31M
```

So **Apple's always-on detector is ~13k–31k weights**. At the 2023 article's 4 bits per weight
that is **~7–15 kB**, the same size class as Google's 13 kB first stage and this chip's
target.

### 2.2 The 2023 system [A2] (stated)

- **First stage:** a DNN-HMM on the Always On Processor, reading a **ring buffer**.
  - 23 output states: 7 phonemes × 3, plus silence and background.
  - Trained end-to-end through a differentiable HMM decoder, so the DNN is tuned for the
    detection score rather than per-frame accuracy.
  - Weights palettised to **4 bits**.
- **Second stage:** a conformer encoder trained with CTC on phonemes, plus a discriminative
  two-class head. Conformers beat BiLSTMs and transformers there.
- **Then:**
  - an LSTM speaker-embedding check;
  - three false-trigger-mitigation models: a lattice RNN on ASR output, an acoustic streaming
    transformer over ~1 s blocks, and a text BERT-like model for non-vocative "Siri".

## 3. Google "Ok Google" / "Hey Google"

### 3.1 The cascade paper [G1] (stated)

**Hardware and budget**
- The first stage runs on a power-isolated DSP.
- The system budget is < 1 mA, i.e. < 1% of battery per day.
- DSPs typically have ~128 kB of memory; program ~25 kB, FFT tables ~12 kB.
- "we have set the footprint of our first stage keyword model to **13kB**."
- Features: log-mel, 32–40 channels, 25 ms frames.

**Audio buffer:** "typically **2 seconds**" of 16 kHz 16-bit audio (64 kB), so the second stage
can re-examine the keyword.

**Second stage:** a larger model on the application processor, plus LSTM speaker
verification. Verification cuts FA by 5–10× for < 1% absolute extra FRR.

**Measured operating points (Table 1).** FA/h on 924 h of TV audio; FRR on 65,581 keyword
recordings:

| stage-1 FA/h | stage-1 FRR | cascade FA/h | cascade FRR |
|---|---|---|---|
| (stage 2 only) | — | 0.03 | 3.1% |
| 0.5 | 4.1% | 0.006 | 5.6% |
| 0.8 | 3.4% | 0.006 | 5.1% |
| 5.0 | 1.6% | 0.02 | 3.8% |
| 10.0 | 1.2% | 0.02 | 3.5% |

**→ Our first-stage target:** about **5 FA/h at ~2% FRR**. Their negatives are TV audio and
ours are conversation plus near-misses, so the numbers aren't directly comparable, but this is
the bar.

### 3.2 End-to-end streaming KWS [G2] (stated)

**Architecture:** SVDF (singular-value-decomposition filter) layers, rank 1. Each node is a
feature filter followed by a short time filter of T past values (T = 8, or 32 in the last
layers), with linear bottlenecks between layers. Inputs are 40 log-mel bands at 10 ms; one
inference every 20 ms.

**Model sizes:**

| model | params | MACs / inference | FRR @ 0.1 FA/h (clean, non-accented) |
|---|---|---|---|
| baseline (conv + 3 FC) | ~1.7M | 1.8M | 1.46% |
| E2E_700K | 700k | 350k | 0.87% |
| E2E_318K | 318k | 159k | 0.87% |
| E2E_40K (1 stage) | 40k | 20k | 1.52% |

The paper also cites a separate **~13k-parameter** always-on first stage.

**Training data:** 1M real "Ok/Hey Google" utterances, each with 8 noise + reverb copies.

### 3.3 Synthetic training data [G3] (stated)

Small keyword models (~400k params) on top of a **pre-trained speech embedding**, trained only
on synthesized speech, match models trained on 500+ real examples. Without the embedding,
~4,000 real examples are needed. This is the closest published evidence for our sim-only
training, and it argues for a **pretrained teacher / distillation** step.

## 4. Amazon Alexa

- **Two-stage on device [Z1] (stated).** A DNN-HMM decoder, then a small feed-forward NN on
  its hypotheses. The second stage cuts false alarms by ~67% "with very few computational
  resources". Monophone background modelling gives −16% FRR at fixed FA.
- **Cloud verification [Z2] (stated).** Each device trigger is re-checked in the cloud on a
  stream of **500 ms pre-roll + the wake word + following speech**. Amazon names its expected
  false wakes, "Alex", "election", "Alexis", and TV ads that say "Alexa"; these are left for
  the cloud to catch.
- **Front end [Z3] (stated).** A trainable DNN front end (time-delayed bottleneck highway
  network) replacing log-mel features.
- **Compressed TDNN** for small-footprint KWS (Sun et al., Interspeech 2017), cited in [Z4];
  size not retrieved.

## 5. Silicon: published always-on KWS chips

| chip | venue | process | front end | classifier | memory | power | accuracy |
|---|---|---|---|---|---|---|---|
| Laika [S1] | ESSCIRC 2018 | 65 nm | external | LSTM accelerator | on-chip | 5 µW | GSCD |
| Vocell [S2] | JSSC 2020 | 65 nm | on-chip, analog mic interface | recurrent (Laika lineage), KWS + speaker ID | on-chip | 6.5–18.3 µW | 90.9% GSCD-12 |
| Shan et al. [S3] | ISSCC 2020 | 28 nm | serial-FFT MFCC | binarized depthwise-separable CNN | binary | 510 nW | 1–2 keywords |
| UltraTrail [S4] | TCAD 2020 | 22FDX | MFCC | TC-ResNet8 (TCN) | distributed SRAM | 8.2 µW | ~93% GSCD |
| Kim et al. [S5] | ISSCC/JSSC 2022 | 65 nm | **16-ch filterbank, 16 ms frame shift** (ring-oscillator, time domain) | **GRU + FC** | on-chip | 23 µW | ~86% GSCD-12 |
| DeltaKWS [S6] | arXiv 2024 / IEEE 2025 | 65 nm | **IIR band-pass filterbank** | **delta-GRU**, temporal sparsity | 24 kB weight SRAM | 36 nJ/decision | 89.5% GSCD-12 |
| ternary KWS [S7] | 2023 | 22 nm | on-chip | ternary-weight NN | 14 kB | 3.8 µW | 10 keywords |
| Syntiant NDP101 [S8] | product | — | on-chip | DNN | **570k params** | 140 µW (vendor) | Alexa wake word |

The two chips closest to our power class ([S5], [S6]) use almost exactly this chip's design:
a **16-band filterbank at 16 ms frames, or an IIR band-pass bank, feeding a GRU**.

## 6. Open source

- **microWakeWord** [O1] (ESPHome, ESP32). Streaming MixConv CNN, int8. Trained on **Piper TTS**
  with room, noise and speed augmentation; the same recipe as ours.
- **openWakeWord.** A frozen Google speech-embedding model plus a tiny per-word head. Too big
  for this chip, but a useful accuracy reference (≈ the [G3] approach).
- **Keyword Transformer (KWT)** [O2]. Transformers for KWS: the smallest is **607k params**, 12
  layers, attending over the whole 1 s window; 97.7–98.6% GSCD-12. The authors note
  transformers need more data. No always-on chip uses one.
- **Hello Edge** [O3] (ARM). MCU study; DS-CNN ~38k params best; GRUs over LSTMs for fewer
  parameters.

## 7. What we take from this, and what we've done about it

| finding | action in this project |
|---|---|
| Cascade: a high-recall first stage + a verifier on buffered audio (all three companies) | Done: **rolling audio buffer**. `src/audio_ring.v` keeps 4 s in PSRAM and streams "WAKE" + 2.05 s pre-roll + live audio to the host over UART (`hw/host/stream_reader.py`). Same pre-roll as Google's 2 s. |
| First-stage operating point ≈ 0.5–10 FA/h at 1–4% FRR (Google Table 1) | Report FRR at 5 / 10 / 30 FA/h and FA/h at 5% / 10% FRR (to do in `gpu_train.py`). Tune for recall; near-misses are the host's job (Amazon's "election" list). |
| First-stage size ~13–31k weights (Apple, derived), 13 kB (Google), 40k E2E (Google) | Our ladder brackets this. The 16-tile middle is 29k–98k depending on architecture. |
| **4-bit weights** (Apple 2023) | Halves QSPI traffic: the 16-tile GRU (98k) fits the bus at 4-bit. Plan: quantisation-aware training at 4 bits. |
| Apple: DNN over 20 stacked frames + phonetic states + HMM decoder | Add an **"apple" architecture**: 5-layer DNN over stacked frames. Phonetic-state targets need alignments; Piper can output phoneme durations (to investigate). |
| Google: SVDF E2E | Add an **"svdf" architecture** to the ladder. |
| Filterbank + GRU is the silicon state of the art ([S5], [S6]) | Our front end matches it; GRU is in the ladder. |
| Synthetic data works with a good embedding / teacher [G3] | Plan: distil from a pretrained speech model; training stays simulator-only. |
| Speaker verification after the trigger (Apple, Google) | Host-side, on the streamed audio; not on chip. |

## 8. Should a preprocessing chip go in front?

Open question. The options:

| option | what it saves on our die | cost / risk |
|---|---|---|
| **Own front end** (current: PDM mic → CIC → 16-band IIR filterbank on chip) | — | ~5 tiles of 16. Proven by [S5], [S6]. Fully open and verified bit-exact. |
| I2S MEMS mic (decimation inside the mic) | the CIC, ~0.5 tile | 3-wire input; a higher-power mic |
| Audio DSP / codec in front | the whole front end, ~5 tiles | mW-class power vs µW; the chip becomes "just a classifier"; not open |
| Analog / acoustic-activity front ends (wake-on-sound MEMS mics, analog ML chips) | could gate everything | vendor parts, not yet evaluated |

The deciding numbers:
- The tile cost of the front end (~5 tiles, 1/3 of an 8×2 budget) vs the classifier
  accuracy those tiles would buy. Notebook 04 measures accuracy vs tiles.
- Power, which we can't measure until silicon or an FPGA run.

## References

- [A1] Apple Machine Learning Research, "Hey Siri: An On-device DNN-powered Voice Trigger for Apple's Personal Assistant," 2017. https://machinelearning.apple.com/research/hey-siri
- [A1s] iMore, "How 'Hey Siri' works" (summary of [A1] with the 5×32 / 5×192 sentences). https://imore.com/how-does-siri-work
- [A2] Apple Machine Learning Research, "Voice Trigger System for Siri," Aug 2023. https://machinelearning.apple.com/research/voice-trigger
- [G1] A. Gruenstein, R. Alvarez, C. Thornton, M. Ghodrat, "A Cascade Architecture for Keyword Spotting on Mobile Devices," arXiv:1712.03603, 2017. https://arxiv.org/abs/1712.03603
- [G2] R. Alvarez, H.-J. Park, "End-to-End Streaming Keyword Spotting," ICASSP 2019, arXiv:1812.02802. https://arxiv.org/abs/1812.02802
- [G3] J. Lin, K. Kilgour, D. Roblek, M. Sharifi, "Training Keyword Spotters with Limited and Synthesized Speech Data," ICASSP 2020, arXiv:2002.01322. https://arxiv.org/abs/2002.01322
- [G4] Google patent US 9,418,656, "Multi-stage hotword detection" (coarse first stage on a DSP). https://patents.google.com/patent/US9418656
- [G5] G. Chen, C. Parada, G. Heigold, "Small-footprint keyword spotting using deep neural networks," ICASSP 2014. https://www.clsp.jhu.edu/wp-content/uploads/2015/11/icassp2014_dnn_hotword.pdf
- [Z1] M. Wu et al., "Monophone-based Background Modeling for Two-stage On-device Wake Word Detection," ICASSP 2018. https://www.amazon.science/publications/monophone-based-background-modeling-for-two-stage-on-device-wake-word-detection
- [Z2] Amazon Alexa Voice Service, "Wake Word Verification." https://developer.amazon.com/docs/alexa-voice-service/enable-cloud-based-wake-word-verification.html ; launch post: https://developer.amazon.com/blogs/alexa/post/b136b3e7-0ba8-4589-aaf9-2a037fc4e9c9/cloud-based-wake-word-verification-improves-alexa-wake-word-accuracy-on-your-avs-products
- [Z3] Amazon Science, "Alexa scientists present two new techniques that improve wake word performance," 2018. https://www.amazon.science/blog/alexa-scientists-present-two-new-techniques-that-improve-wake-word-performance
- [Z4] Citation list in arXiv:2010.06659 (M. Sun et al., "Compressed time delay neural network for small-footprint keyword spotting," Interspeech 2017). https://arxiv.org/pdf/2010.06659
- [S1] J. S. P. Giraldo, M. Verhelst, "Laika: A 5uW Programmable LSTM Accelerator for Always-on Keyword Spotting in 65nm CMOS," ESSCIRC 2018. https://www.researchgate.net/publication/328767095
- [S2] J. S. P. Giraldo et al., "Vocell: A 65-nm Speech-Triggered Wake-Up SoC for 10-µW Keyword Spotting and Speaker Verification," JSSC 55(4), 2020. https://lirias.kuleuven.be/retrieve/567109
- [S3] W. Shan et al., "A 510nW 0.41V Low-Memory Low-Computation Keyword-Spotting Chip Using Serial FFT-Based MFCC and Binarized Depthwise Separable CNN in 28nm CMOS," ISSCC 2020. https://faculty.cqu.edu.cn/yangminhao/en/lwcg/410283/content/223082.htm
- [S4] P. Palomero Bernardo et al., "UltraTrail: A Configurable Ultralow-Power TC-ResNet AI Accelerator for Efficient Keyword Spotting," IEEE TCAD 39(11), 2020. https://embedded.uni-tuebingen.de/publications/palomerobernardo-gerum-frischknecht-luebeck-bringmann-ultratrail-tcn-accelerator-2020/
- [S5] K. Kim et al., "A 23-µW Keyword Spotting IC With Ring-Oscillator-Based Time-Domain Feature Extraction," JSSC 57(11), 2022. https://arxiv.org/abs/2208.00693
- [S6] "DeltaKWS: A 65nm 36nJ/Decision Bio-inspired Temporal-Sparsity-Aware Digital Keyword Spotting IC with 0.6V Near-Threshold SRAM," arXiv:2405.03905. https://arxiv.org/abs/2405.03905
- [S7] "A 3.8µW 10-Keyword Noise-Robust Keyword Spotting Processor Using Symmetric Compressed Ternary-Weight Neural Networks," 2023. https://www.researchgate.net/publication/373728278
- [S8] Syntiant, "Is that your standby power?" (NDP101: 570k params, 140 µW). https://www.edge-ai-vision.com/2021/10/is-that-your-standby-power/
- [O1] ESPHome micro_wake_word. https://esphome.io/components/micro_wake_word.html
- [O2] A. Berg, M. O'Connor, M. Tairum Cruz, "Keyword Transformer: A Self-Attention Model for Keyword Spotting," Interspeech 2021, arXiv:2104.00769 (Table 1: KWT-1 607K, KWT-2 2,394K, KWT-3 5,361K params). https://arxiv.org/abs/2104.00769
- [O3] Y. Zhang et al., "Hello Edge: Keyword Spotting on Microcontrollers," arXiv:1711.07128. https://arxiv.org/abs/1711.07128
