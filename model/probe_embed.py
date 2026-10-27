"""Probe P5: a strong pretrained speech representation + a small head trained on OUR data.

  python model/probe_embed.py embed      # embeddings for windows + streams (CPU, onnx)
  python model/probe_embed.py train      # head on GPU -> data/probe/results/embed_<word>.json

Representation: Google's speech_embedding (openWakeWord's frozen backbone, 96-d every 80 ms
from a 0.775 s log-mel window). Head: MLP over the last L embeddings (L=10, 1.5 s of audio),
trained on the same telescope training windows as the front-end probes (simulated only), early
stopping on the val half of the probe streams, scored on the test half. Compare with the
front-end probes' TCN on chip features (same data, same metric): the gap is what a large
pretrained front end buys.
"""

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import probe_eval as E
import store
from synth_tts import load_candidates

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "data/probe"
EMB = PROBE / "emb"
WORD = "telescope"
STREAMS = ("pos_telescope", "sim_convo", "sim_hard", "real_ami")
L = 10
DEV = "cuda"
_af = {}


def af():
    if "af" not in _af:
        from openwakeword.utils import AudioFeatures

        from synth_tts import _limit_ort_threads

        _limit_ort_threads(1)
        _af["af"] = AudioFeatures(ncpu=1)
    return _af["af"]


def emb_windows(job):
    lo, hi, src = job
    feat = af()
    x = np.asarray(store.load(PROBE / src)["audio"][lo:hi])
    spec = feat._get_melspectrogram(x)  # (n, frames, 32) for a batch
    if spec.ndim == 2:
        spec = spec[None]
    wins = np.stack([spec[:, i : i + 76] for i in range(0, spec.shape[1] - 75, 8)], 1)  # (n, k, 76, 32)
    n, k = wins.shape[:2]
    e = feat.embedding_model.run(None, {"input_1": wins.reshape(n * k, 76, 32, 1).astype(np.float32)})[0]
    return lo, e.reshape(n, k, 96).astype(np.float16)


def emb_chunk(job):
    name, c = job
    s = E.load(name)
    a = np.array(s["audio"][s["chunk"][c] : s["chunk"][c + 1]])
    return name, c, af()._get_embeddings(a).astype(np.float16)


def embed_set(src, dst, n_proc=16):
    """Teacher embeddings (N, 11, 96) of a set of 1.6 s windows (data/probe/<src>)."""
    if store.exists(EMB / dst):
        return
    n = len(store.load(PROBE / src)["audio"])
    out = [None] * n
    jobs = [(i, min(n, i + 256), src) for i in range(0, n, 256)]
    with Pool(n_proc) as pool:
        for lo, e in pool.imap_unordered(emb_windows, jobs):
            for j in range(len(e)):
                out[lo + j] = e[j]
    store.save(EMB / dst, X=np.stack(out))
    print(f"{dst}: {np.stack(out).shape}", file=sys.stderr, flush=True)


def embed(n_proc=16, extra=()):
    EMB.mkdir(parents=True, exist_ok=True)
    embed_set(f"windows_{WORD}", "windows", n_proc)
    for src in extra:
        embed_set(src, src, n_proc)
    for s in STREAMS:
        if store.exists(EMB / s):
            continue
        nc = len(E.load(s)["chunk"]) - 1
        res = [None] * nc
        with Pool(n_proc) as pool:
            for _, c, e in pool.imap_unordered(emb_chunk, [(s, c) for c in range(nc)]):
                res[c] = e
        store.save(EMB / s, X=np.stack(res))
        print(f"{s}: {np.stack(res).shape}", file=sys.stderr, flush=True)


class Head(nn.Module):
    def __init__(self, hidden=128, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(L * 96, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):  # (B, L, 96) -> (B,)
        return self.net(x).squeeze(-1)


def stream_scores(model, Xs, tags_src, half, name):
    s = E.load(name)
    ids = list(E.chunk_ids(s, half))
    sc, tg = [], []
    with torch.no_grad():
        for c in ids:
            e = Xs[c]  # (k, 96)
            w = e.unfold(0, L, 1).transpose(1, 2)  # (k-L+1, L, 96)
            sc.append(model(w.float()).float().cpu().numpy())
            ends = (8 * (np.arange(len(w)) + L - 1) + 75) * 160 + 400
            t10 = s["tag"][s["chunk"][c] // E.TAG_RES : s["chunk"][c + 1] // E.TAG_RES]
            tg.append(t10[np.minimum((ends - 1) // E.TAG_RES, len(t10) - 1)])
    return sc, tg


def evaluate(model, S, k, half, real=False):
    model.eval()
    fs = 0.08
    pos = E.region_max(*stream_scores(model, S["pos_telescope"], None, half, "pos_telescope"))
    negs = [E.Neg(*stream_scores(model, S["sim_convo"], None, half, "sim_convo"), fs),
            E.Neg(*stream_scores(model, S["sim_hard"], None, half, "sim_hard"), fs, own=(k,))]
    reals = [E.Neg(*stream_scores(model, S["real_ami"], None, "all", "real_ami"), fs)] if real else []
    model.train()
    r = E.report(pos, negs, reals)
    r["pos_max"] = [round(float(x), 5) for x in pos]  # per keyword region, in pos_*_meta.json order (this half)
    return r


def train(seed=0, steps=4000, bs=512, lr=1e-3, hidden=128, drop=0.0, wd=1e-2, val_every=500, out_tag="", emb_dir=None):
    torch.manual_seed(seed)
    emb = emb_dir or EMB
    w = store.load(PROBE / f"windows_{WORD}")
    tag, contains = np.asarray(w["tag"]), np.asarray(w["contains"])
    k = [c["phrase"] for c in load_candidates()].index(WORD)
    X = torch.from_numpy(np.asarray(store.load(emb / "windows")["X"])).to(DEV)  # (N, 11, 96)
    clean = ~((contains[:, k // 8] >> (7 - k % 8)) & 1 == 1)
    pos = torch.tensor(np.nonzero(tag == k)[0], device=DEV)
    neg = torch.tensor(np.nonzero((tag != k) & clean)[0], device=DEV)
    S = {s: torch.from_numpy(np.asarray(store.load(emb / s)["X"])).to(DEV) for s in STREAMS}
    model = Head(hidden, drop).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * s / steps)))
    K = X.shape[1]
    best, curve, t0 = (9.0, None, 0), [], time.time()
    for step in range(1, steps + 1):
        ip = pos[torch.randint(len(pos), (bs // 2,), device=DEV)]
        ineg = neg[torch.randint(len(neg), (bs // 2,), device=DEV)]
        off = torch.randint(0, K - L + 1, (bs // 2,), device=DEV)  # negatives: any position
        xp = X[ip, K - L :]
        xn = X[ineg[:, None], off[:, None] + torch.arange(L, device=DEV)]
        x = torch.cat([xp, xn]).float()
        x = x + 0.05 * torch.randn_like(x)
        y = torch.cat([torch.ones(bs // 2, device=DEV), torch.zeros(bs // 2, device=DEV)])
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % val_every == 0:
            r = evaluate(model, S, k, "val")
            score = float(np.mean([r["frr@1.0"], r["frr@2.0"], r["frr@5.0"]]))
            curve.append({"step": step, "val_score": score, "loss": loss.item()})
            print(f"step {step} val {score:.3f}", file=sys.stderr, flush=True)
            if score < best[0]:
                best = (score, {a: b.clone() for a, b in model.state_dict().items()}, step)
    model.load_state_dict(best[1])
    res = {"probe": f"P5 speech_embedding + MLP head ({emb.name})", "seed": seed, "hidden": hidden, "drop": drop, "lr": lr, "wd": wd, "best_step": best[2], "val_score": best[0],
           "curve": curve, "head_params": sum(p.numel() for p in model.parameters()), "train_s": round(time.time() - t0)}
    res.update(evaluate(model, S, k, "test", real=True))
    out = PROBE / "results" / f"embed_{WORD}{out_tag}_s{seed}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({a: b for a, b in res.items() if a != "curve"}), file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["embed", "train"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--tag", default="")
    ap.add_argument("--extra", default="", help="more window sets to embed, e.g. multi_train,windows_computer")
    args = ap.parse_args()
    if args.what == "embed":
        embed(args.jobs, [e for e in args.extra.split(",") if e])
    else:
        train(args.seed, args.steps, lr=args.lr, hidden=args.hidden, drop=args.drop, wd=args.wd,
              val_every=args.val_every, out_tag=args.tag)
