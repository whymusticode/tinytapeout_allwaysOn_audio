"""Precompute feature windows for every candidate word, hard negative and background.

One shared cache serves all 105 candidates: each window carries a tag, and
training for word k uses tag == k as positives and everything else as negatives.

Training data is 100% simulated (mix.Mixer("train")): TTS speech, recorded ambience,
simulated rooms. The test cache has held-out simulated positives and real recordings.

Tags:  0..K-1  candidate k (TTS)
       -1      ambience only (sometimes with crowd babble)
       -2      partial-phrase hard negative (TTS), e.g. "hey", "little penguin"
       -3      conversation (TTS, 1-3 talkers)
       2000+k  phonetic near-miss of candidate k (TTS), e.g. "hey ducks" for "hey tux"
       1000+k  real recording of candidate k (Speech Commands or MSWC), test only

contains: packed bits, which candidates occur in the window's conversation speech.
Training for word k never uses a window with bit k set as a negative.

Output: data/cache/{train,test}/ (store.py format)
        X (N, T, 16) uint8 feature codes, tag (N,) int32, contains (N, ceil(K/8)) uint8

Memory: inputs are memory-mapped (shared by all workers) and each worker writes its
windows straight into the preallocated output file, so RAM use stays flat no
matter how big the cache is.
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import mix
import store
from frontend import N_BANDS
from synth_tts import load_candidates, slug

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
REAL_SETS = ("sc_test", "mswc_test")  # real single-word recordings
REAL_PER_WORD = 1500
N_CONVO = {"train": 400_000, "test": 0}
# Training mixes per clip (each mix: new room, distance, noise, speed, preceding speech).
# With 2 mixes per keyword clip the detector memorised its ~4k positives within ~3k steps.
MIXES = {"pos": 16, "near": 8, "partial": 2}
N_NOISE = {"train": 60_000, "test": 0}

_mixers = {}
_out = None


def _init(split):
    global _out
    _out = {k: store.open_rw(CACHE / split, k) for k in ("X", "contains")}


def _mixer(mode):
    if mode not in _mixers:
        _mixers[mode] = mix.Mixer(mode)
    return _mixers[mode]


def _work(job):
    path, lo, hi, aug, seed, mode, kind, at = job
    rng = np.random.default_rng(seed)
    m = _mixer(mode)
    kw = {}
    j = at
    if path is None:
        for _ in range(hi - lo):
            _out["X"][j], _out["contains"][j] = m.window(rng, kind)
            j += 1
    else:
        d = store.load(path)
        audio, off = d["audio"], d["offsets"]
        for i in range(lo, hi):
            clip = audio[off[i] : off[i + 1]].astype(np.float32) / 32768
            for _ in range(aug):
                _out["X"][j], _out["contains"][j] = m.window(rng, "clip", clip, **kw)
                j += 1
    _out["X"].flush()
    _out["contains"].flush()
    return j - at


def jobs_for(split, chunk=200):
    """-> jobs, tags. Each job knows where its windows go in the output (field `at`)."""
    jobs, tags = [], []
    seed = 0 if split == "train" else 10_000_000
    sim = "train" if split == "train" else "simtest"
    mixes = MIXES if split == "train" else {k: 1 for k in MIXES}

    def add(path, lo, hi, tag, aug, mode=sim, kind="clip"):
        nonlocal seed
        jobs.append((str(path) if path else None, lo, hi, aug, seed, mode, kind, len(tags)))
        tags.extend([tag] * ((hi - lo) * aug))
        seed += 1

    def add_set(path, n, tag, aug, **kw):
        for lo in range(0, n, chunk):
            add(path, lo, min(n, lo + chunk), tag, aug, **kw)

    for k, c in enumerate(load_candidates()):
        p = ROOT / "data" / "synth" / slug(c["phrase"]) / split
        add_set(p, len(store.load(p)["offsets"]) - 1, k, mixes["pos"])
    for p in sorted((ROOT / "data" / "synth_neg").glob(f"*/{split}")):
        add_set(p, len(store.load(p)["offsets"]) - 1, -2, mixes["partial"])
    for k, c in enumerate(load_candidates()):
        p = ROOT / "data" / "synth_near" / slug(c["phrase"]) / split
        if store.exists(p):
            add_set(p, len(store.load(p)["offsets"]) - 1, 2000 + k, mixes["near"])
    add_set(None, N_CONVO[split], -3, 1, kind="convo")
    add_set(None, N_NOISE[split], -1, 1, kind="noise")
    if split == "test":  # real recordings of the candidates, at a distance in real rooms with real ambience
        names = [c["phrase"] for c in load_candidates()]
        for name in REAL_SETS:
            path = ROOT / "data" / "pools" / name
            words = store.load(path)["word"]
            # clips are stored in word order, so each word is one contiguous range
            for k, w in enumerate(names):
                idx = np.nonzero(words == w)[0][:REAL_PER_WORD]
                for lo in range(idx[0], idx[-1] + 1, chunk) if len(idx) else ():
                    add(path, lo, min(idx[-1] + 1, lo + chunk), 1000 + k, 1, mode="test")
    return jobs, np.array(tags, dtype=np.int32)


def build(split, n_jobs):
    jobs, tags = jobs_for(split)
    print(f"{split}: {len(tags)} windows, {len(tags) * mix.T * N_BANDS / 1e9:.2f} GB", file=sys.stderr)
    store.create(CACHE / split, "X", (len(tags), mix.T, N_BANDS), np.uint8)
    store.create(CACHE / split, "contains", (len(tags), mix.N_CONTAINS), np.uint8)
    # Big jobs first so the pool doesn't end on one straggler.
    order = sorted(range(len(jobs)), key=lambda i: -(jobs[i][2] - jobs[i][1]) * jobs[i][3])
    done = 0
    with Pool(n_jobs, initializer=_init, initargs=(split,), maxtasksperchild=50) as pool:
        for i, n in enumerate(pool.imap_unordered(_work, [jobs[i] for i in order])):
            done += n
            if i % 200 == 0:
                print(f"{split}: {i}/{len(jobs)} jobs, {done} windows", file=sys.stderr, flush=True)
    assert done == len(tags), (done, len(tags))
    store.save(CACHE / split, tag=tags)
    print(f"{split}: {len(tags)} windows saved " + str(dict(zip(*np.unique(np.where(tags >= 2000, 2000, np.minimum(tags, 1000)), return_counts=True)))),
          file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=30)
    ap.add_argument("--split", default="test,train")
    args = ap.parse_args()
    mix.features(np.zeros(mix.WIN))  # compile numba once (cached on disk) before forking workers
    for split in args.split.split(","):
        build(split, args.jobs)
