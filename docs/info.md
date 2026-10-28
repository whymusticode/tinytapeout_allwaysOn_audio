<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

This is the first piece of an open-source always-on wake-word detector: a small chip that listens to a
microphone all the time and wakes a host CPU (a Linux phone, smart speaker, handheld, ...) when it hears a
keyword.

This tapeout contains the audio front end:

1. **PDM microphone interface.** With `ui[6]` (PDM_EXT) low, the chip clocks the mic on `uo[1]`
   (clk / 16 = 1.024 MHz at 16.384 MHz). With `ui[6]` high it listens in on a mic that something
   else clocks (for example a phone's codec): the mic clock comes in on `ui[1]`, up to clk / 5
   (3.2 MHz). PDM data is on `ui[0]`; `ui[7]` (PDM_EDGE) selects which half of the mic clock carries
   our channel.
2. **3rd-order CIC decimator.** Decimation R is a register (default 64: 1.024 MHz -> 16 kHz; use 96,
   128, 150, 192 for 1.536, 2.048, 2.4, 3.072 MHz mic clocks), output 16-bit PCM.
3. **Energy VAD.** A one-pole envelope follower (about a 4 ms time constant) on |PCM|. `uo[3]` is
   high while the envelope is above the VAD_THR register (default 1024 of 32767).
4. **Rolling audio buffer for the host.** Every PCM sample goes into a 4-second ring buffer in PSRAM A
   on the QSPI Pmod (`uio`). While the host holds `ui[2]` (STREAM_REQ) high, `uo[2]` (AUDIO_TX, UART
   8N1 at 1,024,000 baud) sends `WAKE`, then int16 little-endian 16 kHz PCM, starting 2.05 s before
   the request and continuing live until `ui[2]` drops. `uo[4]` is high during a stream. This is the
   cascade pattern the big voice assistants use: the always-on stage keeps a short history (Google's
   DSP stage buffers 2 s) for a larger second-stage check on the host.
5. **Filterbank.** 12 mel-spaced bands (150 Hz - 7 kHz), 4th-order resonators, log2 energy every
   16 ms as 6-bit codes: the keyword model's input. The latest frame is readable over SPI.
6. **Host SPI** (`ui[3]` CS_N, `ui[4]` SCK, `ui[5]` MOSI, `uo[5]` MISO; mode 0, SCK <= clk / 16):
   configuration, status, the feature frame, and PCM injection (the host feeds audio in place of
   the mic, to test the filterbank on its own). `uo[7:6]` show two internal signals picked by a
   register. Register map: `src/spi_cfg.v`.

Planned next: the streaming keyword classifier (weights in the Pmod flash) and the wake IRQ on
`uo[0]`.

## How to test

1. Run the chip at 16.384 MHz. The audio buffer needs at least 12 MHz (PSRAM refresh limit).
2. Connect a PDM MEMS microphone: CLK to `uo[1]`, DATA to `ui[0]`, `ui[6]` = 0, and `ui[7]` to match
   the mic's L/R select (0 for a mic that drives data while CLK is high).
3. Speak: `uo[3]` (VAD) goes high while you talk. Over SPI, read register 0x00 (expect 0x57) and
   0x15 (PCM sample counter, should advance) to check the chip and the mic.
4. Plug the TT QSPI Pmod into the `uio` header. Wire `uo[2]` to a 3.3 V UART RX at 1,024,000 baud,
   and `ui[2]` to a GPIO. Holding `ui[2]` high streams the last 2 s plus live audio
   (`hw/host/stream_reader.py` in the repo does this from a Raspberry Pi and saves a WAV).

## External hardware

A PDM MEMS microphone breakout (for example any 1.8–3.3 V PDM mic with CLK/DATA/SEL pins) wired to the
pins above. For the audio buffer: the Tiny Tapeout QSPI Pmod (flash + 2× APS6404L PSRAM) on `uio`, and
a UART receiver (for example a Raspberry Pi) on `uo[2]`/`ui[2]`.
