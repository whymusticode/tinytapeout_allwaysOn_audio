"""Write example audio to listen to: what the simulator makes vs the real test material.

  python model/examples.py      # -> data/examples/{sim,real}/*.wav + data/examples/INDEX.md

Scenes are 6-10 s (longer than a 1.6 s training window) so they can be judged by ear.
Filenames carry the scene parameters (keyword, distance, muffled or not).
All files are 16 kHz mono, at the level the chip would see.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

import mix
import store
from synth_tts import slug, trim

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "examples"
RATE = mix.RATE


def norm(x, dbfs=-26):
    x = x * (10 ** (dbfs / 20) / mix.rms(x))
    return np.clip(x, -1, 1).astype(np.float32)


def keyword_scene(m, rng, clip, n_s=7.0, at_s=4.0, talk=True, snr_ref=15, d=None):
    """Ambience (+ people talking) with the keyword said at a distance, ending at at_s seconds."""
    n = int(n_s * RATE)
    bg, _ = m.background(rng, n)
    x = bg / mix.rms(bg) * 10 ** (-snr_ref / 20)  # a talker at REF_M has rms 1
    if talk:
        conv, _ = m.conversation(rng, n)
        conv[int((at_s - 1.8) * RATE) : int((at_s + 0.3) * RATE)] = 0  # leave a gap for the keyword
        x = x + conv
    p = m.placement(rng, d)
    c = trim(clip.astype(np.float32), RATE, pad_s=0.05)
    c = m.place(c / mix.rms(c), p)
    end = int(at_s * RATE)
    x[max(0, end - len(c)) : end] += c[-end:]
    muff = "muffled" if p["sos"] is not None and _cutoff(p) < 2100 else "open"
    return norm(x), f"d{p['d']:.1f}m_{muff}"


def _cutoff(p):
    from scipy.signal import sosfreqz

    w, h = sosfreqz(p["sos"], worN=512, fs=RATE)
    return w[np.argmax(np.abs(h) < 0.707)]


def clip_from(path, i):
    d = store.load(path)
    return np.asarray(d["audio"][d["offsets"][i] : d["offsets"][i + 1]], np.float32) / 32768


def write(sub, name, x, index, note):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
    sf.write(OUT / sub / f"{name}.wav", x, RATE)
    index.append(f"| `{sub}/{name}.wav` | {note} |")


def main():
    rng = np.random.default_rng(42)
    index = ["# Example audio", "", "| file | what |", "|---|---|"]
    sim, real = mix.Mixer("train"), mix.Mixer("test")

    # --- simulator ---
    for i in range(3):
        x = sim.stream(rng, 10 * RATE)
        write("sim", f"conversation_{i}", norm(x), index, "TTS conversation (1-3 talkers, distances, rooms) over ambience")
    for i in range(2):
        b, _ = sim.babble(rng, 8 * RATE)
        a = sim.ambience(rng, 8 * RATE)
        write("sim", f"crowd_{i}", norm(a / mix.rms(a) + b / mix.rms(b)), index, "TTS crowd babble over ambience")
    for word in ("telescope", "alchemist", "house"):
        p = ROOT / "data/synth" / slug(word) / "train"
        if not store.exists(p):
            continue
        n = len(store.load(p)["offsets"]) - 1
        for d in (0.5, 2.0, 4.5):
            x, tag = keyword_scene(sim, rng, clip_from(p, rng.integers(n)), d=d)
            write("sim", f"keyword_{word}_{tag}", x, index, f"TTS '{word}' at {d} m (ends at 4.0 s), people talking before")
        x, tag = keyword_scene(sim, rng, clip_from(p, rng.integers(n)), talk=False, snr_ref=30, d=0.3)
        write("sim", f"keyword_{word}_clean", x, index, f"TTS '{word}', close and quiet (voice reference)")
        q = ROOT / "data/synth_near" / slug(word) / "train"
        if store.exists(q):
            x, tag = keyword_scene(sim, rng, clip_from(q, rng.integers(len(store.load(q)["offsets"]) - 1)), d=1.0)
            write("sim", f"nearmiss_of_{word}_{tag}", x, index, f"TTS near-miss of '{word}' (must NOT trigger)")

    # --- real (test only) ---
    ami = store.load(ROOT / "data/pools/ami_test") if store.exists(ROOT / "data/pools/ami_test") else None
    if ami is None:  # pools not rebuilt yet: read the meetings directly
        files = sorted((ROOT / "data/raw/ami").glob("*.Array1-01.wav"))
        for i, f in enumerate(rng.choice(files, 3, replace=False)):
            info = sf.info(str(f))
            s = int(rng.integers(0, info.frames - 15 * RATE))
            x, _ = sf.read(str(f), start=s, frames=15 * RATE, dtype="float32")
            write("real", f"ami_meeting_{i}_{f.name.split('.')[0]}", x, index,
                  "AMI meeting, far-field table mic, untouched (true level; turn the volume up)")
    real_words = []
    for word in ("telescope", "alchemist", "house"):
        fs = sorted((ROOT / "data/raw/mswc").glob(f"*/{word}_common_voice_*.opus"))
        if fs:
            real_words.append((word, fs))
    for word, fs in real_words:
        for d in (0.5, 2.0, 4.5):
            a, sr = sf.read(str(fs[rng.integers(len(fs))]), dtype="float32")
            from scipy.signal import resample_poly

            a = resample_poly(a, RATE, sr).astype(np.float32)
            x, tag = keyword_scene(real, rng, a, d=d)
            write("real", f"keyword_{word}_{tag}", x, index,
                  f"real speaker (Common Voice) '{word}' placed at {d} m in a real recorded room, real talker + ambience")
        a, sr = sf.read(str(fs[rng.integers(len(fs))]), dtype="float32")
        sf.write(OUT / "real" / f"keyword_{word}_as_recorded.wav", a, sr)
        index.append(f"| `real/keyword_{word}_as_recorded.wav` | real '{word}' exactly as recorded (voice reference) |")
    lib = ROOT / "data/pools/libri_test"
    if store.exists(lib):
        n = len(store.load(lib)["offsets"]) - 1
        c = clip_from(lib, rng.integers(n))[: 9 * RATE]
        x, tag = keyword_scene(real, rng, c, n_s=10, at_s=9.5, talk=False, d=1.5)
        write("real", f"read_speech_{tag}", x, index, "real read speech (LibriSpeech) at 1.5 m in a real room + ambience")
    (OUT / "INDEX.md").write_text("\n".join(index) + "\n")
    print("\n".join(index))


if __name__ == "__main__":
    main()
