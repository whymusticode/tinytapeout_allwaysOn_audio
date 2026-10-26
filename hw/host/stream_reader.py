#!/usr/bin/env python3
"""Host side of the chip's rolling audio buffer: request a stream, save it as a WAV.

The chip keeps the last ~4 s of its mic audio in PSRAM. While STREAM_REQ (ui[2]) is held
high it sends, on AUDIO_TX (uo[2], UART 8N1 at 1,024,000 baud): "WAKE", then int16
little-endian 16 kHz PCM starting 2.05 s before the request, then live audio.

Raspberry Pi wiring (chip at 3.3 V I/O):
  chip uo[2] AUDIO_TX   -> Pi GPIO15 (RXD, header pin 10)
  chip ui[2] STREAM_REQ <- Pi GPIO23 (header pin 16)
  chip uo[0] WAKE_IRQ   -> Pi GPIO24 (header pin 18), once the classifier exists
  GND                   -- Pi GND

  python3 stream_reader.py --seconds 4 --out grab.wav            # grab now (last 2 s + 2 s live)
  python3 stream_reader.py --on-wake --seconds 4 --out-dir wakes # second-stage loop

Needs pyserial and gpiozero (both on Raspberry Pi OS); enable the UART with
`dtparam=uart0=on` in /boot/firmware/config.txt (and no serial console on it).
"""

import argparse
import time
import wave
from pathlib import Path

RATE = 16000
BAUD = 1_024_000
MAGIC = b"WAKE"


def grab(ser, req, seconds):
    """Hold STREAM_REQ for the pre-roll + `seconds` of live audio; return int16 bytes."""
    ser.reset_input_buffer()
    req.on()
    want = len(MAGIC) + int((2.05 + seconds) * RATE) * 2
    buf = bytearray()
    t_end = time.time() + seconds + 2.0  # the pre-roll drains ~3x faster than real time
    while len(buf) < want and time.time() < t_end + 1.0:
        buf += ser.read(max(1, min(4096, want - len(buf))))
        if time.time() > t_end:
            req.off()
    req.off()
    i = buf.find(MAGIC)
    if i < 0:
        raise RuntimeError("no WAKE header: check wiring, baud rate and that the chip is out of reset")
    pcm = bytes(buf[i + len(MAGIC) :])
    return pcm[: len(pcm) // 2 * 2]


def save(path, pcm):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


def main():
    import serial
    from gpiozero import Button, OutputDevice

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyAMA0")
    ap.add_argument("--req-gpio", type=int, default=23)
    ap.add_argument("--wake-gpio", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=2.0, help="live audio after the pre-roll")
    ap.add_argument("--on-wake", action="store_true", help="wait for WAKE_IRQ, grab, repeat")
    ap.add_argument("--out", default="grab.wav")
    ap.add_argument("--out-dir", default="wakes")
    args = ap.parse_args()

    ser = serial.Serial(args.port, BAUD, timeout=0.2)
    req = OutputDevice(args.req_gpio, initial_value=False)
    if not args.on_wake:
        pcm = grab(ser, req, args.seconds)
        save(args.out, pcm)
        print(f"{args.out}: {len(pcm) // 2 / RATE:.2f} s")
        return
    wake = Button(args.wake_gpio, pull_up=False)
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    print("waiting for WAKE_IRQ ...")
    while True:
        wake.wait_for_press()
        pcm = grab(ser, req, args.seconds)
        name = out / f"wake_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        save(name, pcm)
        print(f"{name}: {len(pcm) // 2 / RATE:.2f} s -> run the second-stage verifier on it")


if __name__ == "__main__":
    main()
