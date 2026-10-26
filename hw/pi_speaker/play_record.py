#!/usr/bin/env python3
"""Run on the Raspberry Pi 5: play stimuli.wav through the PAM8302 amp + speaker, and record
it at the same time: with the PDM mics (record.py running, --rec pdm), a USB mic, or a phone.

Standard library + gpiozero only (both ship with Raspberry Pi OS); audio via aplay/arecord.

  python3 play_record.py devices                       # list playback / capture devices
  python3 play_record.py level --play plughw:2,0       # loop pink noise to set the amp's volume pot
  python3 play_record.py run --play plughw:2,0 --rec pdm --dist 2.0 --dist1 0.5 --note "living room"
  python3 play_record.py run --play plughw:2,0 --rec plughw:3,0 --dist 2.0      # USB mic instead

`run` writes recordings/<label>.wav + .json (distance, notes, devices); with --rec pdm one
pair per mic (<label>_mic0, <label>_mic1), cut from the always-on recorder's files.
The amp is shut down (SD pin low) except while playing, so it can't hiss into the recording
between takes.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDM_REC = Path.home() / "recordings"  # record.py --out
SD_GPIO = 17  # BCM numbering: GPIO17 = header pin 11


class Amp:
    """PAM8302 SD pin: high = on, low = shutdown. Falls back to always-on without gpiozero."""

    def __init__(self, gpio=SD_GPIO):
        try:
            from gpiozero import OutputDevice

            self.pin = OutputDevice(gpio, active_high=True, initial_value=False)
        except Exception as e:  # not on a Pi, or SD left unconnected
            print(f"(no amp shutdown control: {e})", file=sys.stderr)
            self.pin = None

    def on(self):
        if self.pin:
            self.pin.on()
            time.sleep(0.05)  # let the amp's output settle before audio starts

    def off(self):
        if self.pin:
            self.pin.off()


def devices(_):
    for cmd in (["aplay", "-l"], ["arecord", "-l"]):
        print("$", " ".join(cmd))
        subprocess.run(cmd)


def level(args):
    amp = Amp()
    amp.on()
    print("Looping pink noise. Turn the amp's pot so a phone SPL meter at 0.5 m reads ~65-70 dB(A)"
          " (a normal speaking voice). Ctrl-C to stop.")
    try:
        while True:
            subprocess.run(["aplay", "-q", "-D", args.play, str(HERE / "out/pink.wav")], check=True)
    except KeyboardInterrupt:
        pass
    finally:
        amp.off()


def run(args):
    stim = HERE / "out/stimuli.wav"
    rec_dir = HERE / "recordings"
    rec_dir.mkdir(exist_ok=True)
    label = args.label or f"{datetime.now():%Y%m%d_%H%M%S}_d{args.dist:g}m"
    meta = {"label": label, "dist_m": args.dist, "note": args.note, "play": args.play, "rec": args.rec,
            "spl_dba_at_0.5m": args.spl, "time": datetime.now().isoformat(timespec="seconds")}
    amp = Amp()
    rec = None
    if args.rec and args.rec != "pdm":
        # 16 kHz mono; plughw converts from whatever the USB mic runs at. Stops on its own.
        dur = int(_wav_seconds(stim)) + 3
        rec = subprocess.Popen(["arecord", "-q", "-D", args.rec, "-f", "S16_LE", "-r", "16000", "-c", "1",
                                "-d", str(dur), str(rec_dir / f"{label}.wav")])
        time.sleep(1.0)
    amp.on()
    t_start = time.time()
    try:
        print(f"playing {stim.name} ({_wav_seconds(stim):.0f} s) ...")
        subprocess.run(["aplay", "-q", "-D", args.play, str(stim)], check=True)
    finally:
        amp.off()
    t_end = time.time()
    if args.rec == "pdm":
        _cut_pdm(args, label, meta, t_start, t_end, rec_dir)
        return
    if rec:
        rec.wait()
        print(f"recorded {rec_dir / (label + '.wav')}")
    else:
        print("no --rec device: copy your phone recording to", rec_dir / f"{label}.wav")
    (rec_dir / f"{label}.json").write_text(json.dumps(meta, indent=1))


def _cut_pdm(args, label, meta, t0, t1, rec_dir):
    """Close the recorder's current files (SIGUSR1), then cut this take out for each mic."""
    import wave

    sys.path.insert(0, str(HERE.parent / "pi_mics"))
    from record import extract

    os.kill(int((PDM_REC / "recorder.pid").read_text()), signal.SIGUSR1)  # needs sudo if it runs as a service
    time.sleep(2.0)
    for mic, d in ((0, args.dist), (1, args.dist1)):
        if d is None:
            continue
        x = extract(t0 - 1.0, t1 + 1.0, mic, PDM_REC)
        if len(x) == 0:
            print(f"mic{mic}: nothing recorded in that time - is record.py running?")
            continue
        name = f"{label}_mic{mic}"
        with wave.open(str(rec_dir / f"{name}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((x.clip(-1, 1) * 32767).astype("<i2").tobytes())
        (rec_dir / f"{name}.json").write_text(json.dumps({**meta, "label": name, "mic": f"pdm{mic}", "dist_m": d}, indent=1))
        print(f"recorded {rec_dir / (name + '.wav')} ({len(x) / 16000:.0f} s, {d} m)")


def _wav_seconds(path):
    import wave

    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices").set_defaults(fn=devices)
    p = sub.add_parser("level")
    p.add_argument("--play", required=True, help="ALSA playback device, e.g. plughw:2,0")
    p.set_defaults(fn=level)
    p = sub.add_parser("run")
    p.add_argument("--play", required=True)
    p.add_argument("--rec", help="'pdm' (both PDM mics via record.py), an ALSA capture device (USB mic), "
                                 "or omit when recording with a phone")
    p.add_argument("--dist", type=float, required=True, help="speaker-to-mic distance, m (mic 0 with --rec pdm)")
    p.add_argument("--dist1", type=float, help="speaker-to-mic-1 distance, m (--rec pdm)")
    p.add_argument("--spl", type=float, help="SPL meter reading dB(A) at 0.5 m during `level`")
    p.add_argument("--note", default="", help="room, placement, 'muffled: mic in pocket', ...")
    p.add_argument("--label")
    p.set_defaults(fn=run)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
