"""Build the playback file for the speaker experiment (run on the workstation, in the repo env).

  python hw/pi_speaker/make_stimuli.py [--word telescope] [--out hw/pi_speaker/out]

-> out/stimuli.wav (16 kHz mono int16, copy this to the Pi) and out/manifest.json.

Layout of stimuli.wav:
  sync sweep | 1 kHz tone | pink noise | clips (keyword, near-misses, conversation) | sync sweep
The two sweeps let analyze.py find the clips in a recording made by any device and correct
for the recorder's clock running slightly fast or slow.
Every clip is normalised to the same RMS, so level differences in a recording come only
from the room, the distance and the speaker/mic.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
import store  # noqa: E402
from synth_tts import trim  # noqa: E402

RATE = 16000
CLIP_DBFS = -20  # RMS of every clip
GAP_S = 1.2  # silence after each clip (lets the room tail die away)


def sweep(rate=RATE, dur=1.0, f0=100, f1=7000):
    """Exponential sine sweep, faded in and out."""
    t = np.arange(int(dur * rate)) / rate
    k = np.log(f1 / f0)
    x = np.sin(2 * np.pi * f0 * dur / k * (np.exp(t / dur * k) - 1))
    fade = np.minimum(1, np.minimum(t, dur - t) / 0.02)
    return (x * fade * 0.5).astype(np.float32)


def pink(n, rng):
    f = np.fft.rfftfreq(n, 1 / RATE)
    spec = (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) / np.sqrt(np.maximum(f, 50))
    x = np.fft.irfft(spec, n)
    return (x / np.sqrt(np.mean(x**2)) * 10 ** (CLIP_DBFS / 20)).astype(np.float32)


def clips_from(path, n, rng, max_s=4.0):
    d = store.load(path)
    off = np.asarray(d["offsets"])
    out = []
    for i in rng.choice(len(off) - 1, size=min(n, len(off) - 1), replace=False):
        a = trim(np.asarray(d["audio"][off[i] : off[i + 1]], np.float32) / 32768, RATE, pad_s=0.05)
        if len(a) <= max_s * RATE:
            out.append((int(i), a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="telescope")
    ap.add_argument("--n_key", type=int, default=30)
    ap.add_argument("--n_near", type=int, default=10)
    ap.add_argument("--n_convo", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "hw/pi_speaker/out"))
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    parts, items = [], []
    pos = 0

    def add(x, kind, **info):
        nonlocal pos
        if kind not in ("sync", "gap"):
            items.append({"kind": kind, "start": pos, "len": len(x), **info})
        parts.append(x.astype(np.float32))
        pos += len(x)

    def gap(s):
        add(np.zeros(int(s * RATE), np.float32), "gap")

    gap(0.5)
    add(sweep(), "sync")
    gap(1.0)
    t = np.arange(3 * RATE) / RATE
    add((np.sin(2 * np.pi * 1000 * t) * np.sqrt(2) * 10 ** (CLIP_DBFS / 20)).astype(np.float32), "tone_1k")
    gap(1.0)
    add(pink(5 * RATE, rng), "pink")
    gap(1.0)
    groups = [
        ("keyword", ROOT / "data/synth" / args.word / "test", args.n_key),
        ("nearmiss", ROOT / "data/synth_near" / args.word / "test", args.n_near),
        ("convo", ROOT / "data/synth_convo/test", args.n_convo),
    ]
    clips = [(kind, i, a) for kind, path, n in groups for i, a in clips_from(path, n, rng)]
    for j in rng.permutation(len(clips)):
        kind, i, a = clips[j]
        add(a / np.sqrt(np.mean(a**2)) * 10 ** (CLIP_DBFS / 20), kind, src=i)
        gap(GAP_S)
    sync_end = pos
    add(sweep(), "sync")
    gap(0.5)

    x = np.clip(np.concatenate(parts), -1, 1)
    sf.write(out / "stimuli.wav", x, RATE, subtype="PCM_16")
    sf.write(out / "pink.wav", pink(10 * RATE, rng), RATE, subtype="PCM_16")  # for setting the volume
    manifest = {"rate": RATE, "word": args.word, "clip_dbfs": CLIP_DBFS, "sync_start": int(0.5 * RATE),
                "sync_end": sync_end, "items": items}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    kinds = {k: sum(it["kind"] == k for it in items) for k in ("keyword", "nearmiss", "convo")}
    print(f"{out / 'stimuli.wav'}: {len(x) / RATE:.0f} s, {kinds}")


if __name__ == "__main__":
    main()
