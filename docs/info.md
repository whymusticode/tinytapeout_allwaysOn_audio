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

1. **PDM microphone interface.** The chip generates the mic clock on `uo[1]` (clk / 16, which is 1.024 MHz
   at the nominal 16.384 MHz clock) and reads 1-bit PDM data from `ui[0]`. `ui[1]` selects which half of the
   clock the mic drives data in. For a stereo pair this also picks the left or right channel.
2. **3rd-order CIC decimator.** It decimates by 64, turning 1.024 MHz PDM into 16 kHz, 16-bit PCM.
3. **Energy VAD.** A one-pole envelope follower (about a 4 ms time constant) on |PCM|. `uo[3]` goes high
   while the envelope is above the threshold set on `ui[7:5]` (threshold = code × 512, out of 32767).

4. **Rolling audio buffer for the host.** The chip writes every PCM sample into a 4-second ring buffer
   in PSRAM A on the QSPI Pmod (`uio`). While the host holds `ui[2]` (STREAM_REQ) high, `uo[2]`
   (AUDIO_TX, UART 8N1 at 1,024,000 baud) sends `WAKE`, then int16 little-endian 16 kHz PCM. The
   audio starts 2.05 s before the request and continues live until `ui[2]` drops. `uo[4]` is high
   during a stream.

   This is the cascade pattern the big voice assistants use. Their always-on detector keeps a
   short history (Google's DSP stage buffers 2 s) for a larger second-stage check on the host.
   It also makes the chip a standalone recorder.

Planned next blocks are a 16-band IIR filterbank (designed and verified, not yet in this
tapeout), the streaming keyword classifier, and the wake IRQ on `uo[0]`.

## How to test

1. Run the chip at 16.384 MHz. Other clocks scale every rate proportionally.
2. Connect a PDM MEMS microphone: CLK to `uo[1]`, DATA to `ui[0]`, and the mic's L/R select pin to match
   `ui[1]`. For a mic that drives data while CLK is high, set `ui[1]` = 0.
3. Set a VAD threshold on `ui[7:5]`, for example 2, then speak. `uo[3]` should go high while you talk.
4. Plug the TT QSPI Pmod into the `uio` header. Wire `uo[2]` to a 3.3 V UART RX at 1,024,000 baud, and
   `ui[2]` to a GPIO. Holding `ui[2]` high streams the last 2 s plus live audio
   (`hw/host/stream_reader.py` in the repo does this from a Raspberry Pi and saves a WAV).

## External hardware

A PDM MEMS microphone breakout (for example any 1.8–3.3 V PDM mic with CLK/DATA/SEL pins) wired to the
pins above. For the audio buffer: the Tiny Tapeout QSPI Pmod (flash + 2× APS6404L PSRAM) on `uio`, and
a UART receiver (for example a Raspberry Pi) on `uo[2]`/`ui[2]`.
