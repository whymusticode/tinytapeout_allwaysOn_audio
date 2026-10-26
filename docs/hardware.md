# Bring-up and data-collection hardware

## Parts list

| Item | Qty | Why | Where |
|---|---|---|---|
| Adafruit PDM MEMS Microphone Breakout #3492 (ST MP34DT01-M, 1.8–3.3 V, pins 3V/GND/SEL/CLK/DAT) | 3–4 | The same interface the chip uses (raw PDM). Spares let us test a stereo pair. | Adafruit, Amazon, DigiKey, Mouser |
| TT QSPI Pmod (W25Q128JV 16 MB flash + 2× APS6404L 8 MB PSRAM) | 1 | Weight storage and scratch RAM. Uses the uio Pmod pinout already reserved in `info.yaml`. | store.tinytapeout.com |
| TT FPGA Development Kit (iCE40UP5K breakout + ETR demoboard, €90) | 1 | Runs this RTL before tapeout. The template's `fpga.yaml` builds for it. | store.tinytapeout.com |
| Female-female Dupont jumpers, short | 1 pack | Mic to Pmod wiring | anywhere |

**Don't buy INMP441 or SPH0645 boards.** They are the most common "MEMS mic" listings on Amazon, but they
output I2S, not PDM, so they won't connect to our front end.

## Data collection

- **Phones are fine for crowd-sourced keyword and background data right away.** Record mono WAV at 16 kHz or
  higher. If the app allows it, turn off noise suppression and AGC; on Android, use an app that records with the
  `UNPROCESSED` or `VOICE_RECOGNITION` source.
- **For data that matches the target hardware, capture raw PDM from the MP34DT01 mic.** The RP2350 on the
  demoboard can do this with PIO. Running those captures through `model/frontend.py` gives exactly the
  features the chip will see.

## External memory power notes

- **Flash (W25Q128JV)** keeps the weights with no power. It can sit in deep power-down (about 1 µA) until the
  VAD fires. That makes it the right place to store weights for an always-on design.
- **PSRAM (APS6404L)** has to stay powered to keep its contents. The datasheet gives about 100 µA typical
  standby at 25 °C. Use it for scratch and audio buffering only while the system is awake.
- Streaming a model of about 10k ternary weights (about 2.5 KB) every 16 ms frame needs about 160 KB/s. That is
  about 2% duty at 16 MHz quad-SPI. Gated by the VAD, the average cost is dominated by speech time.
