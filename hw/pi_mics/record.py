#!/usr/bin/env python3
"""Always-on recorder: both PDM mics, continuously, as the chip would hear them.

  sudo .venv/bin/python record.py --out ~/recordings          # Ctrl-C to stop
  sudo .venv/bin/python record.py --out ~/recordings --raw    # also keep the raw PDM bits

Runs pdm_capture, decodes each mic with the chip's own CIC (pdm.py, bit-exact to
pdm_cic.v), and writes 16 kHz int16 FLAC in 10-minute files:
  <out>/<YYYY-MM-DD>/<HHMMSS>_mic<m>.flac   + .json (start time, mic, PDM edge, sample count)
--raw also writes <HHMMSS>_mic<m>.pdm: the mic's 1.024 MHz bits, packed 8 per byte, MSB first
(~0.46 GB/h per mic), so later front-end changes can be replayed on real mic bits.

SIGUSR1 closes the current files right away (so a take that just ended can be cut out).
Every 10 s it logs each mic's level, and checks the sample count against the wall clock
(drift = audio lost if the reader fell behind).

extract(t0, t1, mic, out_dir) cuts a stretch of wall-clock time back out of the files
(used by ../pi_speaker/play_record.py and for labelled sessions).
"""

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

import pdm

HERE = Path(__file__).resolve().parent
BLOCK = 128_000  # bytes per read: 0.25 s of capture
WORDS_PER_S = pdm.PDM_HZ * 4 // 32  # 4 bits per PDM clock


def read_exact(f, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = f.readinto(view[got:])
        if not k:
            return None
        got += k
    return bytes(buf)


class MicFile:
    """Rotating FLAC (+ optional raw .pdm) writer for one mic."""

    def __init__(self, out, mic, edge, minutes, raw):
        self.out, self.mic, self.edge, self.raw = out, mic, edge, raw
        self.max_samples = minutes * 60 * pdm.RATE
        self.f = None
        self.last = np.zeros(1)

    def _open(self, t):
        day = self.out / datetime.fromtimestamp(t).strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        stem = day / f"{datetime.fromtimestamp(t):%H%M%S}_mic{self.mic}"
        self.f = sf.SoundFile(str(stem) + ".flac", "w", pdm.RATE, 1, "PCM_16", format="FLAC")
        self.pdm_f = open(str(stem) + ".pdm", "wb") if self.raw else None
        self.meta = {"mic": self.mic, "start_unix": t, "rate": pdm.RATE, "pdm_hz": pdm.PDM_HZ,
                     "pdm_edge": self.edge, "decimator": "cic3 R=64 (pdm_cic.v)", "samples": 0}
        self.meta_path = Path(str(stem) + ".json")
        self.n = 0

    def write(self, pcm, bits, t):
        """t: wall-clock time of pcm[0]."""
        if self.f is None or self.n >= self.max_samples:
            self.close()
            self._open(t)
        self.f.write(pcm)
        self.last = pcm
        if self.pdm_f:
            self.pdm_f.write(np.packbits(bits).tobytes())
        self.n += len(pcm)

    def close(self):
        if self.f is not None:
            self.f.close()
            if self.pdm_f:
                self.pdm_f.close()
            self.meta["samples"] = self.n
            self.meta_path.write_text(json.dumps(self.meta, indent=1))
            self.f = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "recordings"))
    ap.add_argument("--minutes", type=int, default=10, help="file length")
    ap.add_argument("--raw", action="store_true", help="also keep raw PDM bits")
    ap.add_argument("--capture", default=str(HERE / "pdm_capture"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "recorder.pid").write_text(str(__import__("os").getpid()))  # for `kill -USR1`

    proc = subprocess.Popen([args.capture], stdout=subprocess.PIPE, bufsize=0)
    stop, rotate = [], []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGUSR1, lambda *_: rotate.append(1))

    # 1 s to find which clock half each mic drives
    first = read_exact(proc.stdout, WORDS_PER_S * 4)
    t0 = time.time() - 1.0 - 4 * 32768 / (WORDS_PER_S * 4)  # minus the capture's DMA buffering
    if first is None:
        sys.exit("pdm_capture produced no data")
    bits = pdm.split(first)
    phase = [pdm.find_phase(bits[m]) for m in (0, 1)]
    for m in (0, 1):
        if phase[m] is None:
            print(f"mic{m}: no PDM signal on DAT{m} (wiring? power? SEL?) - not recording it", flush=True)
        else:
            print(f"mic{m}: found on the {'low' if phase[m] == 0 else 'high'} clock half "
                  f"(chip PDM_EDGE={pdm.phase_to_edge(phase[m])})", flush=True)
    mics = [m for m in (0, 1) if phase[m] is not None]
    if not mics:
        sys.exit("no mics found")
    cic = {m: pdm.Cic3() for m in mics}
    files = {m: MicFile(out, m, pdm.phase_to_edge(phase[m]), args.minutes, args.raw) for m in mics}

    n_out = 0  # PCM samples written per mic
    last_log = time.time()
    raw = first
    try:
        while raw is not None and not stop:
            if rotate:
                rotate.clear()
                for f in files.values():
                    f.close()
            bits = pdm.split(raw)
            t = t0 + n_out / pdm.RATE
            lens = []
            for m in mics:
                b = bits[m, phase[m]]
                pcm = cic[m].run(b)
                files[m].write(pcm, b, t)
                lens.append(len(pcm))
            n_out += lens[0]
            if time.time() - last_log > 10:
                last_log = time.time()
                drift = (last_log - t0) - n_out / pdm.RATE
                lv = "  ".join(f"mic{m} {level_dbfs(files[m].last):6.1f} dBFS" for m in mics)
                print(f"{datetime.now():%H:%M:%S}  {lv}  behind {drift * 1000:5.0f} ms", flush=True)
            raw = read_exact(proc.stdout, BLOCK)
    finally:
        for f in files.values():
            f.close()
        proc.terminate()


def level_dbfs(pcm):
    x = np.asarray(pcm, np.float64) / 32768
    return 20 * np.log10(np.sqrt(np.mean((x - x.mean()) ** 2)) + 1e-9)


def extract(t0, t1, mic, rec_dir):
    """Cut wall-clock time [t0, t1] for one mic out of the rotating files -> float32 audio."""
    parts = []
    for meta_path in sorted(Path(rec_dir).glob(f"*/*_mic{mic}.json")):
        meta = json.loads(meta_path.read_text())
        s, n = meta["start_unix"], meta["samples"]
        e = s + n / meta["rate"]
        if e < t0 or s > t1:
            continue
        a = int(max(0, (t0 - s) * meta["rate"]))
        b = int(min(n, (t1 - s) * meta["rate"]))
        x, _ = sf.read(str(meta_path.with_suffix(".flac")), start=a, stop=b, dtype="float32")
        parts.append(x)
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


if __name__ == "__main__":
    main()
