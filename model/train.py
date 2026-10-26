"""Train and evaluate a streaming GRU wake-word detector for one candidate.

Model (hardware-shaped): 16 log-band features -> GRU(H) -> linear -> score per 16 ms
frame. Only the GRU state (H values) has to live on chip between frames.

Training data is 100% simulated (TTS + recorded ambience + simulated rooms).

Metrics:
  The threshold is set on the simulated streams (sim_convo + sim_hard, held-out
  speakers / text / ambience / rooms) at 0.5 / 1 / 2 false alarms per hour. At that
  threshold we report:
    frr_syn    missed held-out TTS positives
    frr_real   missed real recordings (Speech Commands control words only)
    fa_<stream> false alarms per hour on each stream, including the real ones
                (real_ami meetings, real_read speech). If the simulator is good,
                real FA/h is close to the sim target, and frr_real close to frr_syn.
  frr_*@real1: the same FRRs with the threshold set on the real streams at 1 FA/h
  (what the detector would actually achieve, calibrated on reality).

Checkpoint selection: every VAL_EVERY steps the model is scored on a validation set of
simulator holdout data (held-out TTS positives + the first VAL_CONVO_CHUNKS minutes of
sim_convo + all of sim_hard) and the best checkpoint by val FRR at VAL_FA FA/h is kept.
Selection never looks at real recordings, so the real numbers stay an unbiased test of the
simulator. frr_real at the val threshold is logged alongside as a monitor only (the
"learning curve" in the results), to see whether sim validation tracks reality.
"""
import copy

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import store
from synth_tts import load_candidates, slug

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
FRAME_S = 0.016
WARMUP = 30  # frames ignored at a window/chunk start while the GRU state settles
POS_LAST = 14  # positives must fire within the last 14 frames of the window
REFRACTORY = 62  # frames (1 s): one false alarm per crossing
FA_TARGETS = (0.5, 1.0, 2.0)
MINE_EVERY, MINE_POOL, MINE_KEEP, MINE_FRAC = 500, 20000, 2000, 0.3  # hard-negative mining
AUG_TILT, AUG_JITTER, AUG_P_MASK = 0.5, 0.25, 0.3  # feature augmentation, log2 units
VAL_EVERY, VAL_FA, VAL_CONVO_CHUNKS = 1000, 1.0, 300  # validation: every 1000 steps, 5 h of sim_convo
SIM_STREAMS = ("sim_convo", "sim_hard")
REAL_STREAMS = ("real_ami", "real_read")


class KWS(nn.Module):
    def __init__(self, n_in=16, hidden=32):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, h=None):
        y, h = self.gru(x, h)
        return self.out(y).squeeze(-1), h


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def load_split(split):
    d = store.load(CACHE / split)  # X is memory-mapped and shared by all sweep processes
    return d["X"], np.array(d["tag"]), np.array(d["contains"])


def has_word(contains, k):
    """contains: (N, ceil(K/8)) packed bits -> bool (N,): window's speech contains candidate k."""
    return (contains[:, k // 8] >> (7 - k % 8)) & 1 == 1


def to_float(codes):
    """uint8 feature codes (1/8 octave) -> float32 log2 units."""
    return np.asarray(codes, dtype=np.float32) / 8


def norm_stats(X, tag):
    """Fixed per-band affine normalisation (a shift + scale in hardware)."""
    idx = np.nonzero(tag == -1)[0][:5000]
    x = to_float(X[idx]).reshape(-1, X.shape[-1])
    return x.mean(0), x.std(0) + 1e-3


def own_tags(k):
    """Tags that count as the target word: its TTS clips and its real recordings (test only)."""
    return {k, 1000 + k}


def augment(x, rng):
    """Feature-space augmentation (log2 units, before normalisation): a random microphone /
    room tilt across bands, per-band jitter, and occasionally a masked band or two.
    Keeps the model from keying on the exact spectrum of the TTS voices."""
    n, _, b = x.shape
    tilt = rng.uniform(-AUG_TILT, AUG_TILT, (n, 1, 1)) * np.linspace(-1, 1, b)[None, None, :]
    x = x + tilt + rng.normal(0, AUG_JITTER, (n, 1, b))
    for i in np.nonzero(rng.random(n) < AUG_P_MASK)[0]:
        x[i, :, rng.integers(b, size=rng.integers(1, 3))] = x[i].mean()
    return x.astype(np.float32)


def train(k, hidden=32, steps=4000, batch=256, seed=0, threads=2, log=None):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cands = load_candidates()
    phrase = cands[k]["phrase"]
    own = own_tags(k)

    X, tag, contains = load_split("train")
    mu, sd = norm_stats(X, tag)
    pos = np.nonzero(tag == k)[0]
    clean = ~has_word(contains, k)  # never call speech that contains the word a negative
    neg_groups = [
        (0.35, np.nonzero((tag == -3) & clean)[0]),  # conversation
        (0.10, np.nonzero(tag == 2000 + k)[0]),  # this word's phonetic near-misses
        (0.25, np.nonzero((tag >= 0) & (tag != k) & (tag != 2000 + k) & clean)[0]),  # other words, their near-misses
        (0.10, np.nonzero((tag == -2) & clean)[0]),  # partial phrases
        (0.20, np.nonzero((tag == -1) & clean)[0]),  # ambience / crowd
    ]
    neg_groups = [(w, g) for w, g in neg_groups if len(g)]
    mu_t, sd_t = torch.tensor(mu), torch.tensor(sd)

    def get(idx, aug=True):
        idx = np.sort(idx)
        x = to_float(X[idx])
        return (torch.tensor(augment(x, rng) if aug else x) - mu_t) / sd_t

    all_neg = np.concatenate([g for _, g in neg_groups])
    hard = None

    @torch.no_grad()
    def mine():
        """Score a random slice of the negatives; the top MINE_KEEP become the hard pool."""
        cand = np.sort(rng.choice(all_neg, MINE_POOL, replace=False))  # sorted, so get() keeps the order
        sc = np.concatenate([model(get(cand[i : i + 4096], aug=False))[0][:, WARMUP:].max(1).values.numpy()
                             for i in range(0, len(cand), 4096)])
        return cand[np.argsort(-sc)[:MINE_KEEP]]

    val = Validator(k, own, (mu, sd))
    best = (np.inf, -1, None)  # (val frr, step, state)
    curve = []
    model = KWS(hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.1)
    bce = nn.BCEWithLogitsLoss()
    n_pos = batch // 4
    t0 = time.time()
    for step in range(steps):
        pi = rng.choice(pos, n_pos)
        n_neg = batch - n_pos
        if step >= MINE_EVERY and step % MINE_EVERY == 0:
            hard = mine()
        n_hard = int(MINE_FRAC * n_neg) if hard is not None else 0
        ni = np.concatenate([rng.choice(g, int(round(w * (n_neg - n_hard)))) for w, g in neg_groups]
                            + ([rng.choice(hard, n_hard)] if n_hard else []))
        xp, xn = get(pi), get(ni)
        lp, _ = model(xp)
        ln, _ = model(xn)
        # positive: fire somewhere in the last frames; negative: never fire after warm-up
        loss_p = bce(lp[:, -POS_LAST:].max(1).values, torch.ones(len(pi)))
        ln = ln[:, WARMUP:]
        loss_n = bce(ln.max(1).values, torch.zeros(len(ni))) + bce(ln, torch.zeros_like(ln))
        loss = loss_p + loss_n
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if log and step % 500 == 0:
            print(f"[{phrase}] step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", file=log, flush=True)
        if (step + 1) % VAL_EVERY == 0 or step == steps - 1:
            v = val(model)
            curve.append({"step": step + 1, "loss": round(loss.item(), 4), **v})
            if v["val_frr"] < best[0]:
                best = (v["val_frr"], step + 1, copy.deepcopy(model.state_dict()))
            if log:
                print(f"[{phrase}] val step {step + 1}: {v}{' *' if best[1] == step + 1 else ''}", file=log, flush=True)
    model.load_state_dict(best[2])
    return model, (mu, sd), phrase, own, {"best_step": best[1], "curve": curve}


class Validator:
    """Simulator-holdout validation, data loaded once. Called with a model -> metrics dict."""

    def __init__(self, k, own, norm):
        self.norm = norm
        Xt, tagt, _ = load_split("test")
        self.Xt = Xt
        self.pos_syn = np.nonzero(tagt == k)[0]
        self.pos_real = np.nonzero(np.isin(tagt, [t for t in own if t >= 1000]))[0]  # monitor only
        self.streams = [load_stream(n, own, VAL_CONVO_CHUNKS if n == "sim_convo" else None) for n in SIM_STREAMS]

    @torch.no_grad()
    def __call__(self, model):
        mu, sd = self.norm
        S = [(stream_logits(model, mu, sd, X, chunk), ign, chunk, hours) for X, ign, chunk, hours in self.streams]
        thr = thr_for(S, VAL_FA)
        res = {"val_frr": float(np.mean(window_scores(model, self.Xt, self.pos_syn, mu, sd, last=POS_LAST) <= thr))}
        if len(self.pos_real):
            res["mon_frr_real"] = float(np.mean(window_scores(model, self.Xt, self.pos_real, mu, sd) <= thr))
        return {k: round(v, 4) for k, v in res.items()}


@torch.no_grad()
def window_scores(model, X, idx, mu, sd, last=None):
    out = []
    for s in range(0, len(idx), 2048):
        x = (torch.tensor(to_float(X[np.sort(idx[s : s + 2048])])) - torch.tensor(mu)) / torch.tensor(sd)
        l, _ = model(x)
        l = l[:, -last:] if last else l[:, WARMUP:]
        out.append(l.max(1).values.numpy())
    return np.concatenate(out) if out else np.zeros(0)


def load_stream(name, own, max_chunks=None):
    """-> (X frames, ignore mask (own word's frames), chunk offsets, hours)."""
    s = store.load(CACHE / f"stream_{name}")
    chunk = np.array(s["chunk"])
    if max_chunks:
        chunk = chunk[: max_chunks + 1]
    X, tag = s["X"][: chunk[-1]], np.array(s["tag"][: chunk[-1]])
    return X, np.isin(tag, list(own)), chunk, len(X) * FRAME_S / 3600


@torch.no_grad()
def stream_logits(model, mu, sd, X, chunk):
    # chunks have different lengths; run in groups of equal length for speed
    lens = np.diff(chunk)
    logits = np.full(len(X), -np.inf, np.float32)
    for L in np.unique(lens):
        ids = np.nonzero(lens == L)[0]
        for s0 in range(0, len(ids), 256):
            g = ids[s0 : s0 + 256]
            x = to_float(np.stack([X[chunk[i] : chunk[i + 1]] for i in g]))
            l, _ = model((torch.tensor(x) - torch.tensor(mu)) / torch.tensor(sd))
            l = l.numpy()
            l[:, :WARMUP] = -np.inf
            for j, i in enumerate(g):
                logits[chunk[i] : chunk[i + 1]] = l[j]
    return logits


def count_fa(logits, ignore, chunk, thr):
    """Rising crossings of thr outside ignored frames, with a 1 s refractory period."""
    idx = np.nonzero((logits > thr) & ~ignore)[0]
    if len(idx) == 0:
        return 0
    cid = np.searchsorted(chunk, idx, side="right")
    new = np.ones(len(idx), bool)
    new[1:] = (np.diff(idx) > REFRACTORY) | (np.diff(cid) != 0)
    return int(new.sum())


def fa_rate(S, thr):
    """S: [(logits, ignore, chunk, hours), ...] -> pooled FA/h at thr."""
    return sum(count_fa(l, i, c, thr) for l, i, c, _ in S) / sum(h for *_, h in S)


def thr_for(S, target):
    """Lowest threshold whose pooled FA/h <= target (bisection on the logit scale)."""
    lo, hi = -20.0, 40.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fa_rate(S, mid) > target else (lo, mid)
    return hi


def evaluate(model, norm, own, k):
    mu, sd = norm
    Xt, tagt, _ = load_split("test")
    pos_syn = window_scores(model, Xt, np.nonzero(tagt == k)[0], mu, sd, last=POS_LAST)
    real_tags = [t for t in own if t >= 1000]
    pos_real = window_scores(model, Xt, np.nonzero(np.isin(tagt, real_tags))[0], mu, sd) if real_tags else None
    S = {}
    for name in SIM_STREAMS + REAL_STREAMS:
        X, ignore, chunk, hours = load_stream(name, own)
        S[name] = (stream_logits(model, mu, sd, X, chunk), ignore, chunk, hours)
    sim, real = [S[n] for n in SIM_STREAMS], [S[n] for n in REAL_STREAMS]

    res = {"stream_hours": {n: round(S[n][3], 2) for n in S}, "n_pos_syn": int(len(pos_syn))}
    if pos_real is not None:
        res["n_pos_real"] = int(len(pos_real))
    for fa in FA_TARGETS:
        thr = thr_for(sim, fa)
        res[f"frr_syn@{fa}"] = float(np.mean(pos_syn <= thr))
        if pos_real is not None:
            res[f"frr_real@{fa}"] = float(np.mean(pos_real <= thr))
        for n in S:
            res[f"fa_{n}@{fa}"] = round(fa_rate([S[n]], thr), 3)
    thr = thr_for(real, 1.0)
    res["frr_syn@real1"] = float(np.mean(pos_syn <= thr))
    if pos_real is not None:
        res["frr_real@real1"] = float(np.mean(pos_real <= thr))
    # which hard-negative content causes the false alarms at the sim 1 FA/h threshold
    thr = thr_for(sim, 1.0)
    logits, ignore = S["sim_hard"][:2]
    hits = np.array(store.load(CACHE / "stream_sim_hard")["tag"])[(logits > thr) & ~ignore]
    res["fa_sources"] = {str(t): int(c) for t, c in zip(*np.unique(hits, return_counts=True))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", required=True, help="slug, e.g. hey_tux")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="data/results/default")
    args = ap.parse_args()
    import sys

    cands = load_candidates()
    k = [slug(c["phrase"]) for c in cands].index(args.word)
    model, norm, phrase, own, hist = train(k, args.hidden, args.steps, threads=args.threads, log=sys.stderr)
    res = {"word": phrase, "slug": args.word, "group": cands[k]["group"], "hidden": args.hidden,
           "params": n_params(model), "best_step": hist["best_step"], **evaluate(model, norm, own, k),
           "curve": hist["curve"]}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.word}.json").write_text(json.dumps(res, indent=1))
    torch.save({"state": model.state_dict(), "mu": norm[0], "sd": norm[1]}, out / f"{args.word}.pt")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
