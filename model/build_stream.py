"""Build the held-out false-alarm streams: hours of continuous audio the detector must ignore.

  sim_convo   simulator, held-out TTS speakers + text + ambience + rooms: 1-3 people talking
  sim_hard    simulator: held-out-speaker clips of every candidate, every partial phrase
              ("hey", "tux", "little penguin", ...) and every candidate's phonetic near-misses
              ("hey ducks"), with pauses, over held-out ambience
  real_ami    AMI meetings, far-field table mic, untouched (real conversation, real rooms)
  real_read   LibriSpeech read speech + Speech Commands words (real voices) placed at a
              distance in real recorded rooms with test ambience (tag -5: real non-candidate word)

The detection threshold is set on the sim streams; the real streams then show whether the
simulator predicts real-world false alarms. Processed in 60 s chunks (the streaming model is
reset at each chunk start). Every frame carries a tag (same scheme as build_cache) so a word's
own clips can be excluded when counting its false alarms.

Output: data/cache/stream_<name>/ (store.py format) with X (frames, 16) uint8 feature codes,
        tag (frames,) int32, chunk (n_chunks + 1,) frame offsets.

Usage: python model/build_stream.py [--only sim_convo,real_ami]
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import mix
import store
from features import HOP, features
from synth_tts import load_candidates, slug

ROOT = Path(__file__).resolve().parent.parent
RATE = 16000
CHUNK = 60 * RATE
TAG_TAIL = RATE // 2  # a clip's tag extends 0.5 s past its end
N_SIM_CONVO = 900  # 15 h

_mixers = {}


def _mixer(mode):
    if mode not in _mixers:
        _mixers[mode] = mix.Mixer(mode)
    return _mixers[mode]


def _convo_chunk(seed):
    rng = np.random.default_rng(seed)
    x = _mixer("simtest").stream(rng, CHUNK)
    np.clip(x, -1, 1, out=x)
    f = features(x)
    return f, np.full(len(f), -3, np.int32)


def _clips_chunk(job):
    """job: (seed, mode, [(audio int16, tag), ...]). Clips end to end with pauses over ambience."""
    seed, mode, clips = job
    m = _mixer(mode)
    rng = np.random.default_rng(seed)
    sig = np.zeros(CHUNK, np.float32)
    tag = np.full(CHUNK, -1, np.int32)
    pos = 0
    for a, t in clips:
        c = a.astype(np.float32) / 32768
        c = m.place(c / mix.rms(c), m.placement(rng))  # at a distance: quieter, more room, duller
        pos += int(rng.uniform(0.1, 1.0) * RATE) if t != -1 else 0
        if pos + len(c) > CHUNK:
            break
        sig[pos : pos + len(c)] = c * 0.1
        if t != -1:
            tag[pos : min(CHUNK, pos + len(c) + TAG_TAIL)] = t
        pos += len(c)
    bg, _ = m.background(rng, CHUNK)
    x = sig + bg * (0.1 / mix.rms(bg) * 10 ** (-rng.uniform(0, 25) / 20))
    x *= mix.db(*mix.LEVEL_DB, rng) / mix.rms(x)
    np.clip(x, -1, 1, out=x)
    f = features(x)
    return f, tag[HOP - 1 :: HOP][: len(f)]  # frame tag = tag at the frame's last sample


def _real_chunk(job):
    path, lo, hi = job
    a = store.load(path)["audio"]
    x = a[lo:hi].astype(np.float32) / 32768  # untouched: real level, real room
    f = features(x)
    return f, np.full(len(f), -1, np.int32)


def clip_list(path, idx, tag):
    d = store.load(path)
    a, off = d["audio"], d["offsets"]
    return [(np.array(a[off[i] : off[i + 1]]), tag) for i in idx]


def pack(clips, rng, max_s=55):
    """Group clips into chunks of at most max_s seconds (gaps included)."""
    rng.shuffle(clips)
    out, cur, dur = [], [], 0.0
    for a, t in clips:
        d = len(a) / RATE + 0.6
        if dur + d > max_s and cur:
            out.append(cur)
            cur, dur = [], 0.0
        cur.append((a, t))
        dur += d
    if cur:
        out.append(cur)
    return out


def sim_hard_jobs(rng):
    clips = []
    for k, c in enumerate(load_candidates()):
        p = ROOT / "data/synth" / slug(c["phrase"]) / "test"
        n = len(store.load(p)["offsets"]) - 1
        clips += clip_list(p, rng.choice(n, size=min(40, n), replace=False), k)
    for p in sorted((ROOT / "data/synth_neg").glob("*/test")):
        n = len(store.load(p)["offsets"]) - 1
        clips += clip_list(p, rng.choice(n, size=min(80, n), replace=False), -2)
    for k, c in enumerate(load_candidates()):
        p = ROOT / "data/synth_near" / slug(c["phrase"]) / "test"
        if store.exists(p):
            n = len(store.load(p)["offsets"]) - 1
            clips += clip_list(p, rng.choice(n, size=min(40, n), replace=False), 2000 + k)
    return [(2000 + i, "simtest", c) for i, c in enumerate(pack(clips, rng))]


def real_read_jobs(rng):
    lib = store.load(ROOT / "data/pools/libri_test")
    la, lo = lib["audio"], lib["offsets"]
    clips = [(np.array(la[lo[i] : lo[i + 1]]), -1) for i in range(len(lo) - 1)]
    sc = store.load(ROOT / "data/pools/sc_test")
    words = sc["word"]
    names = [c["phrase"] for c in load_candidates()]
    take = rng.choice(len(words), size=min(12000, len(words)), replace=False)  # ~3.3 h of real short words
    clips += [(np.array(sc["audio"][sc["offsets"][i] : sc["offsets"][i + 1]]),
               1000 + names.index(str(words[i])) if str(words[i]) in names else -5) for i in take]
    return [(3000 + i, "test", c) for i, c in enumerate(pack(clips, rng))]


def real_ami_jobs():
    path = ROOT / "data/pools/ami_test"
    off = np.array(store.load(path)["offsets"])
    return [(str(path), s, min(s + CHUNK, off[m + 1])) for m in range(len(off) - 1)
            for s in range(off[m], off[m + 1] - RATE * 10, CHUNK)]  # drop tails < 10 s


def build(name, fn, jobs, n_proc=30):
    feats, tags = [], []
    with Pool(n_proc) as pool:
        for i, (f, t) in enumerate(pool.imap(fn, jobs, chunksize=2)):
            feats.append(f)
            tags.append(t)
            if i % 200 == 0:
                print(f"{name}: {i}/{len(jobs)} chunks", file=sys.stderr, flush=True)
    chunk = np.cumsum([0] + [len(f) for f in feats])
    store.save(ROOT / f"data/cache/stream_{name}", X=np.concatenate(feats), tag=np.concatenate(tags), chunk=chunk)
    print(f"stream_{name}: {len(jobs)} chunks, {chunk[-1] * HOP / RATE / 3600:.1f} h", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="sim_convo,sim_hard,real_ami,real_read")
    args = ap.parse_args()
    rng = np.random.default_rng(123)
    mix.features(np.zeros(mix.WIN))  # compile numba before forking
    todo = args.only.split(",")
    if "sim_convo" in todo:
        build("sim_convo", _convo_chunk, list(range(1000, 1000 + N_SIM_CONVO)))
    if "sim_hard" in todo:
        build("sim_hard", _clips_chunk, sim_hard_jobs(rng))
    if "real_ami" in todo:
        build("real_ami", _real_chunk, real_ami_jobs())
    if "real_read" in todo:
        build("real_read", _clips_chunk, real_read_jobs(rng))


if __name__ == "__main__":
    main()
