"""Stream metrics for probes, for any detector that emits one score per frame.

A detector's output on a stream is (scores per frame, frame period in seconds). Frames are
tagged from the stream's 10 ms tags (tag at the frame's last sample).

  FRR at a threshold:  fraction of keyword regions (tag POS) whose max score stays below it
  FA/h at a threshold: upward crossings outside the word's own regions, 1 s refractory, per
                       hour of non-own audio; the first WARM_S of every 60 s chunk is skipped
                       (detector state warming up)
Thresholds for "FRR at x FA/h" are set on the simulated negative streams; real AMI is then
read off at that same threshold (sim -> real transfer) and at its own threshold.
"""

from pathlib import Path

import numba
import numpy as np

import store

ROOT = Path(__file__).resolve().parent.parent
STREAMS = ROOT / "data" / "probe" / "streams"
POS = 9999
TAG_RES = 160
WARM_S = 2.0
FA_TARGETS = (0.5, 1.0, 2.0, 5.0, 10.0)


def load(name):
    d = store.load(STREAMS / name)
    return {"audio": d["audio"], "tag": np.asarray(d["tag"]), "chunk": np.asarray(d["chunk"])}


def chunk_ids(s, half):
    """'val' = even chunks, 'test' = odd chunks, 'all'."""
    n = len(s["chunk"]) - 1
    return {"val": range(0, n, 2), "test": range(1, n, 2), "all": range(n)}[half]


def frame_tags(s, c, n_frames, frame_samples):
    """Tags of chunk c's frames; frame_samples = frame period in 16 kHz samples."""
    t10 = s["tag"][s["chunk"][c] // TAG_RES : s["chunk"][c + 1] // TAG_RES]
    ends = ((np.arange(n_frames) + 1) * frame_samples - 1) // TAG_RES
    return t10[np.minimum(ends, len(t10) - 1)]


@numba.njit(cache=True)
def _count_fa(score, ignore, start, thr, refr):
    n = 0
    last = -(1 << 30)
    for i in range(len(score)):
        if start[i]:
            last = -(1 << 30)
        if ignore[i]:
            continue
        if score[i] > thr and i - last >= refr:
            n += 1
            last = i
    return n


class Neg:
    """Negative stream scores, flattened: score, ignore mask, chunk-start flags."""

    def __init__(self, scores, tags, frame_s, own=()):
        warm = int(WARM_S / frame_s)
        sc, ig, st = [], [], []
        for s, t in zip(scores, tags):
            i = np.isin(t, list(own)) | (t == POS)
            i[:warm] = True
            f = np.zeros(len(s), bool)
            f[0] = True
            sc.append(s.astype(np.float32))
            ig.append(i)
            st.append(f)
        self.score, self.ignore, self.start = np.concatenate(sc), np.concatenate(ig), np.concatenate(st)
        self.hours = (~self.ignore).sum() * frame_s / 3600
        self.refr = int(round(1.0 / frame_s))

    def fa(self, thr):
        return _count_fa(self.score, self.ignore, self.start, thr, self.refr) / self.hours


def fa_rate(negs, thr):
    return sum(_count_fa(n.score, n.ignore, n.start, thr, n.refr) for n in negs) / sum(n.hours for n in negs)


def thr_for(negs, target):
    s = np.concatenate([n.score[~n.ignore] for n in negs])
    lo, hi = float(s.min()) - 1, float(s.max()) + 1
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fa_rate(negs, mid) > target else (lo, mid)
    return hi


def region_max(scores, tags, label=POS):
    """Max score over each keyword region (runs of tag == label) -> array."""
    out = []
    for s, t in zip(scores, tags):
        m = t == label
        if not m.any():
            continue
        edges = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
            out.append(s[a:b].max())
    return np.array(out)


def report(pos_max, sim_negs, real_negs=(), prefix=""):
    """FRR at FA targets (threshold on sim negatives) + real FA at those thresholds."""
    r = {f"{prefix}n_pos": int(len(pos_max)), f"{prefix}sim_hours": round(sum(n.hours for n in sim_negs), 2)}
    for fa in FA_TARGETS:
        thr = thr_for(sim_negs, fa)
        r[f"{prefix}frr@{fa}"] = float(np.mean(pos_max < thr))
        if real_negs:
            r[f"{prefix}real_fa@{fa}"] = round(fa_rate(real_negs, thr), 2)
    if real_negs:
        r[f"{prefix}real_hours"] = round(sum(n.hours for n in real_negs), 2)
        for fa in (1.0, 5.0):
            r[f"{prefix}frr@real{fa}"] = float(np.mean(pos_max < thr_for(real_negs, fa)))
    return r
