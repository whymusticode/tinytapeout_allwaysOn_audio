"""Raw-audio probe sets: the same simulated mixes as the feature caches, kept as audio so any
front end (ours with different bit widths, mel, or someone else's pretrained model) can run on
exactly the same data.

  python model/probe_data.py streams      # evaluation streams (all probes)
  python model/probe_data.py windows      # training windows for one word (front-end probes)
  python model/probe_data.py real         # real-audio windows for Stage A validation / test

data/probe/streams/<name>/   audio int16 (16 kHz), tag int16 per 10 ms, chunk sample offsets
  pos_<slug>     held-out-voice keyword clips at a distance over ambience, 1.5-3.5 s apart
                 (tag POS around each clip + 0.5 s), simulator test mode
  sim_convo      simulated conversation over ambience (same seeds as stream_sim_convo)
  sim_hard       every candidate, partial phrase and near-miss clip (same as stream_sim_hard)
  real_ami       AMI far-field meetings, untouched
data/probe/windows_<slug>/   audio int16 (N, 25600), tag int32, contains: training windows
  (simulator train mode), positives = the word, negatives as in build_cache.

data/probe/real_{val,test}/   audio int16 (N, 25600), src int8 (0 AMI, 1 LibriSpeech, 2 MUSAN):
  real recordings for distillation (teacher vs student on the same audio, no labels needed).
  AMI meetings after the ones real_ami uses, alternating val/test by meeting; LibriSpeech
  dev-clean = val, test-clean = test; MUSAN speech crops split in halves.

Chunks are 60 s; even chunks are the validation half, odd chunks the test half.
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import build_stream as bs
import mix
import store
from synth_tts import load_candidates, slug

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "probe"
RATE, CHUNK, TAG_RES = 16000, 60 * 16000, 160
POS = 9999  # tag of a keyword region in pos_* streams
PROBE_WORDS = ("alexa", "hey jarvis", "hey mycroft", "okay nabu")

_m = {}


def mixer(mode):
    if mode not in _m:
        _m[mode] = mix.Mixer(mode)
    return _m[mode]


def to16(x):
    return np.clip(np.round(np.clip(x, -1, 1) * 32767), -32768, 32767).astype(np.int16)


def convo_chunk(seed):
    x = mixer("simtest").stream(np.random.default_rng(seed), CHUNK)
    return to16(x), np.full(CHUNK // TAG_RES, -3, np.int16)


def clips_chunk(job):
    """Like build_stream._clips_chunk, returning audio. gap: pause range between clips (s)."""
    seed, mode, clips, gap = job
    m = mixer(mode)
    rng = np.random.default_rng(seed)
    sig = np.zeros(CHUNK, np.float32)
    tag = np.full(CHUNK, -1, np.int16)
    pos = int(rng.uniform(*gap) * RATE)
    meta = []
    for a, t in clips:
        c = a.astype(np.float32) / 32768
        p = m.placement(rng)
        c = m.place(c / mix.rms(c), p)
        if pos + len(c) > CHUNK:
            break
        sig[pos : pos + len(c)] = c * 0.1
        tag[pos : min(CHUNK, pos + len(c) + bs.TAG_TAIL)] = t
        meta.append({"start_s": pos / RATE, "dist_m": round(float(p["d"]), 2), "muffled": bool(p["muffled"]),
                     "room": p["room_kind"], "lowpass_hz": None if p["fc"] is None else float(p["fc"])})
        pos += len(c) + int(rng.uniform(*gap) * RATE)
    bg, _ = m.background(rng, CHUNK)
    snr = rng.uniform(0, 25)
    x = sig + bg * (0.1 / mix.rms(bg) * 10 ** (-snr / 20))
    lv = mix.db(*mix.LEVEL_DB, rng)
    x *= lv / mix.rms(x)
    for d in meta:  # SNR of the clip (speech placed at 0.1 rms before distance loss) vs ambience
        d["snr_db_at_0_5m"] = round(snr, 1)
        d["level_dbfs"] = round(20 * np.log10(lv), 1)
    if getattr(clips_chunk, "want_meta", False):
        return meta
    return to16(x), tag[TAG_RES - 1 :: TAG_RES]


def ami_chunk(job):
    path, lo, hi = job
    a = np.array(store.load(path)["audio"][lo:hi])
    a = np.pad(a, (0, CHUNK - len(a)))[:CHUNK]
    return a, np.full(CHUNK // TAG_RES, -1, np.int16)


def save_stream(name, fn, jobs, n_proc):
    audio, tags = [], []
    with Pool(n_proc) as pool:
        for a, t in pool.imap(fn, jobs, chunksize=1):
            audio.append(a)
            tags.append(t)
    store.save(OUT / "streams" / name, audio=np.concatenate(audio), tag=np.concatenate(tags),
               chunk=np.arange(len(audio) + 1) * CHUNK)
    print(f"{name}: {len(audio)} chunks ({len(audio) / 60:.1f} h)", file=sys.stderr, flush=True)


def pos_jobs(path, seed, max_s=55):
    d = store.load(path)
    a, off = d["audio"], d["offsets"]
    clips = [(np.array(a[off[i] : off[i + 1]]), POS) for i in range(len(off) - 1)]
    rng = np.random.default_rng(seed)
    rng.shuffle(clips)
    jobs, cur, dur = [], [], 0.0
    for c in clips:
        d_s = len(c[0]) / RATE + 2.5
        if dur + d_s > max_s and cur:
            jobs.append(cur)
            cur, dur = [], 0.0
        cur.append(c)
        dur += d_s
    jobs.append(cur)
    return [(seed + i, "simtest", c, (1.5, 3.5)) for i, c in enumerate(jobs)]


def streams(n_proc, convo_h, ami_h):
    words = [(w, OUT / "synth" / slug(w) / "test") for w in PROBE_WORDS]
    words.append(("telescope", ROOT / "data/synth/telescope/test"))
    for i, (w, p) in enumerate(words):
        if not store.exists(OUT / "streams" / f"pos_{slug(w)}"):
            save_stream(f"pos_{slug(w)}", clips_chunk, pos_jobs(p, 50_000 + 1000 * i), n_proc)
    if not store.exists(OUT / "streams" / "sim_convo"):
        save_stream("sim_convo", convo_chunk, list(range(1000, 1000 + int(convo_h * 60))), n_proc)
    if not store.exists(OUT / "streams" / "sim_hard"):
        jobs = [(s, m, c, (0.1, 1.0)) for s, m, c in bs.sim_hard_jobs(np.random.default_rng(123))]
        save_stream("sim_hard", clips_chunk, jobs, n_proc)
    if not store.exists(OUT / "streams" / "real_ami"):
        save_stream("real_ami", ami_chunk, bs.real_ami_jobs()[: int(ami_h * 60)], n_proc)


# --- training windows for one word ------------------------------------------------------

def win_job(job):
    seed, kind, path, idx, n_mix = job
    rng = np.random.default_rng(seed)
    m = mixer("train")
    xs, cs = [], []
    if path is None:
        for _ in range(idx):
            x, c = m.window(rng, kind, audio=True)
            xs.append(to16(x))
            cs.append(c)
    else:
        d = store.load(path)
        a, off = d["audio"], d["offsets"]
        for i in idx:
            clip = a[off[i] : off[i + 1]].astype(np.float32) / 32768
            for _ in range(n_mix):
                x, c = m.window(rng, "clip", clip, audio=True)
                xs.append(to16(x))
                cs.append(c)
    return np.stack(xs), np.stack(cs)


def windows(word, n_proc, scale=1.0):
    names = [c["phrase"] for c in load_candidates()]
    k = names.index(word)
    rng = np.random.default_rng(77)
    jobs, tags = [], []
    seed = [5_000_000]

    def add(path, idx, tag, n_mix=1, chunk=100):
        for lo in range(0, len(idx), chunk):
            part = idx[lo : lo + chunk]
            jobs.append((seed[0], "clip", str(path), part, n_mix))
            tags.extend([tag] * (len(part) * n_mix))
            seed[0] += 1

    def n_of(p):
        return len(store.load(p)["offsets"]) - 1

    p = ROOT / "data/synth" / slug(word) / "train"
    add(p, np.arange(n_of(p)), k, n_mix=max(1, int(6 * scale)))
    p = ROOT / "data/synth_near" / slug(word) / "train"
    if store.exists(p):
        add(p, np.arange(n_of(p)), 2000 + k, n_mix=2)
    neg = sorted((ROOT / "data/synth_neg").glob("*/train"))
    for p in rng.choice(neg, size=min(30, len(neg)), replace=False):
        add(p, rng.choice(n_of(p), size=min(int(100 * scale), n_of(p)), replace=False), -2)
    for j, w in enumerate(names):
        if j != k:
            p = ROOT / "data/synth" / slug(w) / "train"
            add(p, rng.choice(n_of(p), size=int(100 * scale), replace=False), j)
    for _ in range(int(240 * scale)):  # 100 windows per job
        jobs.append((seed[0], "convo", None, 100, 1))
        tags.extend([-3] * 100)
        seed[0] += 1
    for _ in range(int(60 * scale)):
        jobs.append((seed[0], "noise", None, 100, 1))
        tags.extend([-1] * 100)
        seed[0] += 1
    X = np.zeros((len(tags), mix.WIN), np.int16)
    C = np.zeros((len(tags), mix.N_CONTAINS), np.uint8)
    at = np.cumsum([0] + [len(j[3]) * j[4] if j[2] else j[3] for j in jobs])
    with Pool(n_proc) as pool:
        for i, (x, c) in enumerate(pool.imap(win_job, jobs, chunksize=1)):
            X[at[i] : at[i + 1]], C[at[i] : at[i + 1]] = x, c
            if i % 100 == 0:
                print(f"windows {word}: {i}/{len(jobs)} jobs", file=sys.stderr, flush=True)
    store.save(OUT / f"windows_{slug(word)}", audio=X, tag=np.array(tags, np.int32), contains=C)
    print(f"windows_{slug(word)}: {len(tags)} windows, {X.nbytes / 1e9:.1f} GB", file=sys.stderr)


def multi_job(job):
    seed, mode, path, idx = job
    rng = np.random.default_rng(seed)
    m = mixer(mode)
    d = store.load(path)
    a, off = d["audio"], d["offsets"]
    return np.stack([to16(m.window(rng, "clip", a[off[i] : off[i + 1]].astype(np.float32) / 32768, audio=True)[0])
                     for i in idx])


def multi(n_proc, per_word_train=300, per_word_test=100):
    """All candidates as a word-classification set: train = train voices, simulator train mode;
    test = held-out voices, simulator test mode (rooms/ambience held out too)."""
    names = [c["phrase"] for c in load_candidates()]
    rng = np.random.default_rng(99)
    for split, mode, n in (("train", "train", per_word_train), ("test", "simtest", per_word_test)):
        jobs, labels = [], []
        for k, w in enumerate(names):
            p = ROOT / "data/synth" / slug(w) / split
            nk = len(store.load(p)["offsets"]) - 1
            idx = rng.choice(nk, size=min(n, nk), replace=False)
            for lo in range(0, len(idx), 50):
                jobs.append((6_000_000 + len(jobs) + (0 if split == "train" else 500_000), mode, str(p), idx[lo : lo + 50]))
                labels += [k] * len(idx[lo : lo + 50])
        with Pool(n_proc) as pool:
            X = np.concatenate(pool.map(multi_job, jobs))
        store.save(OUT / f"multi_{split}", audio=X, tag=np.array(labels, np.int32))
        print(f"multi_{split}: {X.shape}", file=sys.stderr, flush=True)


def _meta_job(job):
    clips_chunk.want_meta = True
    return clips_chunk(job)


def metas():
    """Per-clip conditions of the pos_* streams (same seeds, so the same clips), in region order."""
    import json

    words = [(w, OUT / "synth" / slug(w) / "test") for w in PROBE_WORDS]
    words.append(("telescope", ROOT / "data/synth/telescope/test"))
    for i, (w, p) in enumerate(words):
        with Pool(16) as pool:
            ms = pool.map(_meta_job, pos_jobs(p, 50_000 + 1000 * i))
        rows = [dict(chunk=c, **d) for c, m in enumerate(ms) for d in m]
        (OUT / "streams" / f"pos_{slug(w)}_meta.json").write_text(json.dumps(rows))
        print(f"pos_{slug(w)}: {len(rows)} clips", file=sys.stderr)


def real(n_ami=6000, n_libri=3000, seed=7):
    """1.6 s windows of real recordings for Stage A (see module docstring)."""
    rng = np.random.default_rng(seed)
    pools = ROOT / "data/pools"
    W = mix.WIN
    ami = store.load(pools / "ami_test")
    off = np.asarray(ami["offsets"])
    used = {s for _, s, _ in bs.real_ami_jobs()[: 8 * 60]}  # real_ami (Stage B false alarms)
    first_free = next(m for m in range(len(off) - 1) if not any(off[m] <= s < off[m + 1] for s in used) and off[m] > max(used))
    meetings = list(range(first_free, len(off) - 1))
    lib = store.load(pools / "libri_test")
    loff = np.asarray(lib["offsets"])
    n_dev = len(list((ROOT / "data/raw/librispeech_dev").glob("LibriSpeech/**/*.flac")))  # dev first in the pool
    mus = np.asarray(store.load(pools / "speech_test")["audio"])
    for split, part in (("val", 0), ("test", 1)):
        X, src = [], []
        ms = meetings[part::2]
        for _ in range(n_ami):  # whole-meeting audio, silence included (far-field, as on the device)
            m = ms[rng.integers(len(ms))]
            s = rng.integers(off[m], off[m + 1] - W)
            X.append(np.asarray(ami["audio"][s : s + W]))
            src.append(0)
        utts = range(0, n_dev) if part == 0 else range(n_dev, len(loff) - 1)
        for u in rng.choice(list(utts), n_libri):
            a = np.asarray(lib["audio"][loff[u] : loff[u + 1]])
            s = rng.integers(0, max(1, len(a) - W))
            X.append(np.pad(a[s : s + W], (0, max(0, W - len(a)))))
            src.append(1)
        half = len(mus) // 2
        for c in range(half * part, half * (part + 1)):
            for s in (0, mus.shape[1] - W):
                X.append(mus[c, s : s + W])
                src.append(2)
        store.save(OUT / f"real_{split}", audio=np.stack(X).astype(np.int16), src=np.array(src, np.int8))
        print(f"real_{split}: {len(X)} windows; AMI meetings {[str(ami['meeting'][m]) for m in ms][:3]}.. ({len(ms)})",
              file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["streams", "windows", "meta", "multi", "real"])
    ap.add_argument("--word", default="telescope")
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--convo-h", type=float, default=6)
    ap.add_argument("--ami-h", type=float, default=8)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    if args.what == "meta":
        metas()
    elif args.what == "multi":
        multi(args.jobs)
    elif args.what == "real":
        real()
    elif args.what == "streams":
        streams(args.jobs, args.convo_h, args.ami_h)
    else:
        windows(args.word, args.jobs, args.scale)
