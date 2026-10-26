"""Collect background noise, music, speech and room impulse responses into in-memory pools.

Training is simulator-only: TTS speech + recorded ambience + simulated rooms.
Real speech is used only for testing.

  ESC-50        folds 1-4 train, fold 5 test
  DEMAND        environments split by name (TEST_ENVS are test only)
  MUSAN         noise/music split by file hash (1 in 5 -> test); speech -> test only
  RIRs          simulated rooms split by room (1 in 5 -> "simtest"), real recorded rooms -> test
  LibriSpeech   dev-clean + test-clean -> test only
  Speech Cmds   all splits -> test only (real recordings; nothing real is trained on)
  MSWC          real single-word clips of the candidates (Common Voice) -> test only
  AMI           far-field meetings -> test only (real conversation)

Output: data/pools/<name>/ (store.py format): int16 crops at 16 kHz + category ids,
        Speech Commands and LibriSpeech clip sets (audio + offsets)
"""

import hashlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import store

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "pools"
RATE = 16000
CROP = 4 * RATE
TEST_ENVS = {"PCAFETER", "STRAFFIC", "TMETRO", "DLIVING", "NPARK"}


def is_test(path):
    key = f"{path.parent.name}/{path.name}" if path.name.startswith("Room") else path.name
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 5 == 0


def load(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x[:, 0]
    if sr != RATE:
        g = np.gcd(sr, RATE)
        x = resample_poly(x, RATE // g, sr // g).astype(np.float32)
    return x


def to_i16(x):
    return np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)


def load_i16(path):
    return to_i16(load(path))


def catalog():
    """-> {pool_name: [(category, path), ...]}"""
    c = {"bg_train": [], "bg_test": [], "speech_test": [], "rir_train": [], "rir_simtest": [], "rir_test": []}
    sc = RAW / "speech_commands" / "_background_noise_"
    c["bg_train"] += [("sc_noise", p) for p in sorted(sc.glob("*.wav"))]
    for p in sorted((RAW / "esc50").glob("*/audio/*.wav")):
        c["bg_test" if p.name.startswith("5-") else "bg_train"].append(("esc50", p))
    for kind in ("noise", "music"):
        for p in sorted((RAW / "musan").glob(f"musan/{kind}/**/*.wav")):
            c["bg_test" if is_test(p) else "bg_train"].append((f"musan_{kind}", p))
    for p in sorted((RAW / "musan").glob("musan/speech/**/*.wav")):
        c["speech_test"].append(("musan_speech", p))
    for p in sorted((RAW / "demand").glob("*/*/ch01.wav")):
        env = p.parent.name
        c["bg_test" if env in TEST_ENVS else "bg_train"].append(("demand", p))
    rn = RAW / "rirs_noises" / "RIRS_NOISES"
    for p in sorted(rn.glob("pointsource_noises/*.wav")):
        c["bg_test" if is_test(p) else "bg_train"].append(("pointsource", p))
    for p in sorted(rn.glob("real_rirs_isotropic_noises/*.wav")):
        if "noise" in p.name:
            c["bg_test" if is_test(p) else "bg_train"].append(("isotropic", p))
        else:
            c["rir_test"].append(("real_rir", p))
    sim = sorted(rn.glob("simulated_rirs/*/*/*.wav"))
    rng = np.random.default_rng(0)
    for split, n in (("rir_train", 4000), ("rir_simtest", 500)):
        cand = [p for p in sim if is_test(p.parent) == (split == "rir_simtest")]  # split by room
        for i in rng.choice(len(cand), size=min(n, len(cand)), replace=False):
            c[split].append(("sim_rir", cand[i]))
    return c


def _crops(job):
    """Load one file and cut `n` random 4 s crops (shorter files are looped)."""
    cat, path, n, seed = job
    try:
        x = load(path)
    except Exception as e:  # a few corrupt files would otherwise kill the whole build
        print(f"skip {path}: {e}", file=sys.stderr)
        return cat, []
    if len(x) < RATE // 4 or np.abs(x).max() < 1e-4:
        return cat, []
    if len(x) < CROP:
        x = np.tile(x, CROP // len(x) + 1)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = rng.integers(0, len(x) - CROP + 1)
        c = x[s : s + CROP]
        out.append(to_i16(c / max(np.abs(c).max(), 1e-4) * 0.5))  # peak-normalise; level is set at mix time
    return cat, out


def _rir(job):
    cat, path = job
    h = load(path)
    k = int(np.argmax(np.abs(h)))
    h = h[max(0, k - 16) : k + RATE]  # direct path near the start, at most 1 s tail
    return h / np.abs(h).max()


def build_crop_pool(name, items, total, pool):
    cats = sorted({c for c, _ in items})
    if not cats:
        store.save(OUT / name, audio=np.zeros((0, CROP), np.int16), cat=np.zeros(0, int), cats=np.array([""]))
        print(f"{name}: EMPTY (dataset missing?)", file=sys.stderr)
        return
    per_cat = total // len(cats)  # equal weight per category, not per hour of audio
    jobs = []
    for ci, cat in enumerate(cats):
        files = [p for c, p in items if c == cat]
        n_each = max(1, int(np.ceil(per_cat / len(files))))
        for i, p in enumerate(files):
            jobs.append((cat, p, n_each, int(hashlib.md5(f"{name}/{cat}/{i}".encode()).hexdigest()[:8], 16)))
    crops, cat_ids = [], []
    for cat, cs in pool.imap_unordered(_crops, jobs, chunksize=4):
        crops += cs
        cat_ids += [cats.index(cat)] * len(cs)
    cat_ids = np.array(cat_ids)
    # trim each category to per_cat
    keep = np.concatenate([np.nonzero(cat_ids == i)[0][:per_cat] for i in range(len(cats))])
    store.save(OUT / name, audio=np.stack([crops[i] for i in keep]), cat=cat_ids[keep], cats=np.array(cats))
    print(f"{name}: {len(keep)} crops, categories {dict(zip(cats, np.bincount(cat_ids[keep])))}", file=sys.stderr)


def build_clipset(name, files, pool, **extra):
    """Load files (resampled to 16 kHz int16) into one memory-mapped clip set, in order."""
    out = OUT / name
    n = [int(np.ceil(sf.info(str(f)).frames * RATE / sf.info(str(f)).samplerate)) + 1 for f in files]
    audio = store.create(out, "audio", (sum(n),), np.int16)
    offsets = [0]
    for a in pool.imap(load_i16, files, chunksize=4):
        audio[offsets[-1] : offsets[-1] + len(a)] = a
        offsets.append(offsets[-1] + len(a))
    audio.flush()
    del audio
    store.save(out, offsets=np.array(offsets), **extra)
    return offsets


def build_speech_commands(pool):
    """Real recordings, test only: every clip of every word."""
    sc = RAW / "speech_commands"
    files = []
    for d in sorted(p for p in sc.iterdir() if p.is_dir() and not p.name.startswith("_")):
        files += [(d.name, f) for f in sorted(d.glob("*.wav"))]
    build_clipset("sc_test", [f for _, f in files], pool, word=np.array([w for w, _ in files]))
    print(f"sc_test: {len(files)} clips", file=sys.stderr)


def build_mswc(pool, per_word=1500):
    """Real single-word clips (Common Voice speakers), test only. At most per_word per word."""
    files = []
    for p in sorted((RAW / "mswc").glob("*/*.opus")):
        files.append((p.name.split("_common_voice")[0], p))
    rng = np.random.default_rng(0)
    by_word = {}
    for w, p in files:
        by_word.setdefault(w, []).append(p)
    keep = []
    for w in sorted(by_word):
        ps = by_word[w]
        keep += [(w, ps[i]) for i in sorted(rng.permutation(len(ps))[:per_word])]
    build_clipset("mswc_test", [p for _, p in keep], pool, word=np.array([w for w, _ in keep]))
    print(f"mswc_test: {len(keep)} clips, {len(by_word)} words", file=sys.stderr)


def build_librispeech(pool):
    files = sorted((RAW / "librispeech_dev").glob("LibriSpeech/**/*.flac")) + sorted(
        (RAW / "librispeech_test").glob("LibriSpeech/**/*.flac"))
    off = build_clipset("libri_test", files, pool)
    print(f"libri_test: {len(files)} utterances, {off[-1] / RATE / 3600:.1f} h", file=sys.stderr)


def build_ami(pool):
    files = sorted((RAW / "ami").glob("*.Array1-01.wav"))
    off = build_clipset("ami_test", files, pool, meeting=np.array([f.name.split(".")[0] for f in files]))
    print(f"ami_test: {len(files)} meetings, {off[-1] / RATE / 3600:.1f} h", file=sys.stderr)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = catalog()
    for k, v in c.items():
        print(f"{k}: {len(v)} files", file=sys.stderr)
    with Pool(28) as pool:
        build_crop_pool("bg_train", c["bg_train"], 16000, pool)
        build_crop_pool("bg_test", c["bg_test"], 4000, pool)
        build_crop_pool("speech_test", c["speech_test"], 1500, pool)
        for split in ("train", "simtest", "test"):
            rirs = list(pool.imap(_rir, c[f"rir_{split}"], chunksize=16))
            lens = np.cumsum([0] + [len(h) for h in rirs])
            store.save(OUT / f"rir_{split}", audio=np.concatenate(rirs).astype(np.float32), offsets=lens)
            print(f"rir_{split}: {len(rirs)} responses", file=sys.stderr)
        build_speech_commands(pool)
        build_mswc(pool)
        build_librispeech(pool)
        build_ami(pool)


if __name__ == "__main__":
    main()
