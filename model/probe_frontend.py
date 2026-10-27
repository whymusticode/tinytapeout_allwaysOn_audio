"""Front-end probes: same data, same small detector, different front ends.

  python model/probe_frontend.py --variants base,pcm12,coef10 --seeds 0,1,2

For each variant (frontend_var.py): features for the telescope training windows and the probe
streams (cached in data/probe/feat/<variant>/), then a fixed TCN (~20k params, receptive field
~1 s, one output per ~32 ms) trained on GPU with early stopping on the streams' val half, and
scored on the test half (probe_eval). Results: data/probe/results/frontend/<variant>_s<seed>.json
with the hardware cost of the variant (filterbank flip-flops, multiplier cycles).
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import frontend_var as FV
import probe_eval as E
import store
from synth_tts import load_candidates

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "data" / "probe"
DEV = "cuda"
WORD = "telescope"
STREAMS = ("pos_telescope", "sim_convo", "sim_hard", "real_ami")
NEG_WEIGHTS = {"convo": 0.35, "near": 0.10, "other": 0.25, "partial": 0.10, "amb": 0.20}

# one change at a time from today's chip (FE_BASE); see frontend_var.py for what each field is
VARIANTS = {
    "base": {},
    "pcm14": dict(pcm_bits=14), "pcm12": dict(pcm_bits=12), "pcm10": dict(pcm_bits=10), "pcm8": dict(pcm_bits=8),
    "coef12": dict(coef_frac=12), "coef10": dict(coef_frac=10), "coef8": dict(coef_frac=8),
    "coef6": dict(coef_frac=6),
    "yw18": dict(y_w=18), "yw16": dict(y_w=16), "yw14": dict(y_w=14), "yw12": dict(y_w=12),
    "guard0": dict(guard=0), "guard2": dict(guard=2),
    "acc20": dict(acc_w=20, acc_shift=6), "acc16": dict(acc_w=16, acc_shift=10), "acc12": dict(acc_w=12, acc_shift=14),
    "log2": dict(log_frac=2), "log1": dict(log_frac=1), "log0": dict(log_frac=0),
    # code bits: 7 at 1/8 octave, 6 at 1/4, 5 at 1/2, 4 at 1 octave, all from log2 energy 7 up
    "feat7": dict(feat_bits=7), "feat6": dict(log_frac=2, feat_bits=6), "feat5": dict(log_frac=1, feat_bits=5),
    "feat4": dict(log_frac=0, feat_bits=4), "feat6f": dict(log_frac=3, feat_bits=6),
    "bands8": dict(bands=8), "bands12": dict(bands=12), "bands24": dict(bands=24), "bands32": dict(bands=32),
    "hop32ms": dict(hop=512), "hop10ms": dict(hop=160), "hop24ms": dict(hop=384),
    "rate8k": dict(rate=8000, hop=128),
    "float16": dict(kind="float_iir"),
    "mel16": dict(kind="mel", bands=16), "mel40": dict(kind="mel", bands=40),
    "mel40_10ms": dict(kind="mel", bands=40, hop=160),
    # where does mel's +3 points come from? float IIR with one change each
    "fpow": dict(kind="float_iir", energy="pow"), "ford4": dict(kind="float_iir", order=4),
    "fq2": dict(kind="float_iir", q_mult=2.0), "fq05": dict(kind="float_iir", q_mult=0.5),
    "ford4pow": dict(kind="float_iir", order=4, energy="pow"),
    "fbands24": dict(kind="float_iir", bands=24), "fq2pow": dict(kind="float_iir", q_mult=2.0, energy="pow"),
    # round 2: combinations of the cuts that cost nothing alone (66-word probe, round 1)
    "lean1": dict(pcm_bits=12, coef_frac=8, y_w=18, log_frac=2, feat_bits=6),
    "lean2": dict(pcm_bits=12, coef_frac=8, y_w=18, log_frac=2, acc_w=20, feat_bits=6),
    "lean3": dict(pcm_bits=12, coef_frac=8, y_w=18, log_frac=2, bands=12, feat_bits=6),
    "lean4": dict(pcm_bits=12, coef_frac=8, y_w=16, guard=2, log_frac=2, feat_bits=6),
    "lean5": dict(pcm_bits=12, coef_frac=8, y_w=18, log_frac=2, acc_w=20, acc_shift=4, feat_bits=6),
    "lean7": dict(pcm_bits=12, coef_frac=8, y_w=16, guard=2, log_frac=2, acc_w=20, acc_shift=4, feat_bits=6),
    "lean8": dict(pcm_bits=12, coef_frac=8, y_w=16, guard=2, log_frac=2, acc_w=18, acc_shift=6, feat_bits=6),
    "lean9": dict(pcm_bits=12, coef_frac=8, y_w=16, guard=2, log_frac=2, acc_w=20, acc_shift=4, feat_bits=6, bands=12),
    # 4th-order bands (two cascaded resonators): the float probe says this is where mel's +2-3 pts come from
    "ord4": dict(order=4),
    "lean5o4": dict(pcm_bits=12, coef_frac=8, y_w=18, log_frac=2, acc_w=20, acc_shift=4, feat_bits=6, order=4),
    "lean5o4b": dict(pcm_bits=12, coef_frac=10, y_w=18, log_frac=2, acc_w=20, acc_shift=4, feat_bits=6, order=4),
    "lean7o4": dict(pcm_bits=12, coef_frac=8, y_w=16, guard=2, log_frac=2, acc_w=20, acc_shift=4, feat_bits=6, order=4),
    "lean5o4b12": dict(pcm_bits=12, coef_frac=8, y_w=18, log_frac=2, acc_w=20, acc_shift=4, feat_bits=6, order=4, bands=12),
    "lean5o4p10": dict(pcm_bits=10, coef_frac=8, y_w=18, log_frac=2, acc_w=20, acc_shift=4, feat_bits=6, order=4),
    "lean5o4b14": dict(pcm_bits=12, coef_frac=8, y_w=18, log_frac=2, acc_w=20, acc_shift=4, feat_bits=6, order=4, bands=14),
    "lean6": dict(pcm_bits=12, coef_frac=8, y_w=18, guard=2, log_frac=2, acc_w=20, acc_shift=2, feat_bits=6),
}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --- features ---------------------------------------------------------------------------

def feat_path(name):
    return PROBE / "feat" / name


def build_features(name, v, batch_chunks=48):
    """Cached features of the training windows and every probe stream for variant v."""
    p = feat_path(name)
    if store.exists(p / "real_ami"):
        return
    t0 = time.time()
    w = store.load(PROBE / f"windows_{WORD}")
    A = w["audio"]
    out = []
    for i in range(0, len(A), 4096):
        out.append(FV.features(np.asarray(A[i : i + 4096]), v))
    store.save(p / "windows", X=np.concatenate(out))
    for s in STREAMS:
        d = E.load(s)
        n = len(d["chunk"]) - 1
        out = []
        for i in range(0, n, batch_chunks):
            a = np.asarray(d["audio"][d["chunk"][i] : d["chunk"][min(n, i + batch_chunks)]]).reshape(-1, E.load(s)["chunk"][1])
            out.append(FV.features(a, v))
        store.save(p / s, X=np.concatenate(out))
    log(f"features {name}: {time.time() - t0:.0f}s")


# --- model ------------------------------------------------------------------------------

class TCN(nn.Module):
    """Frames stacked by `stride` (~32 ms steps), then causal depthwise-separable conv blocks,
    kernel 3, dilations 1,2,4,8 (receptive field 31 steps ~ 1 s), residual."""

    def __init__(self, n_in, stride, width=64, dil=(1, 2, 4, 8)):
        super().__init__()
        self.stride, self.dil = stride, dil
        self.inp = nn.Linear(n_in * stride, width)
        self.dw = nn.ModuleList([nn.Conv1d(width, width, 3, dilation=d, groups=width) for d in dil])
        self.pw = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in dil])
        self.bn = nn.ModuleList([nn.BatchNorm1d(width) for _ in dil])
        self.out = nn.Linear(width, 1)

    def forward(self, x):  # (B, T, C) -> (B, T) logits, held over each stride
        B, T, C = x.shape
        s = self.stride
        pad = (-T) % s
        x = F.pad(x, (0, 0, 0, pad)).reshape(B, (T + pad) // s, s * C)
        h = self.inp(x).transpose(1, 2)
        for d, dw, pw, bn in zip(self.dil, self.dw, self.pw, self.bn):
            h = h + F.relu(bn(pw(dw(F.pad(h, (2 * d, 0))))))
        y = self.out(h.transpose(1, 2)).squeeze(-1)
        if s > 1:  # step j is known after frame j*s+s-1
            y = y.repeat_interleave(s, dim=1)
            y = F.pad(y[:, : y.shape[1] - (s - 1)], (s - 1, 0), value=-20.0)
        return y[:, :T]


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# --- data on GPU ------------------------------------------------------------------------

class Data:
    def __init__(self, name, v):
        p = feat_path(name)
        w = store.load(PROBE / f"windows_{WORD}")
        tag, contains = np.asarray(w["tag"]), np.asarray(w["contains"])
        k = [c["phrase"] for c in load_candidates()].index(WORD)
        self.k = k
        self.fs = FV.frame_s(v)
        X = FV.to_log2(np.asarray(store.load(p / "windows")["X"]), v)
        clean = ~((contains[:, k // 8] >> (7 - k % 8)) & 1 == 1)
        negm = (tag != k) & clean
        self.mu = torch.tensor(X[negm].reshape(-1, X.shape[-1]).mean(0), device=DEV)
        self.sd = torch.tensor(X[negm].reshape(-1, X.shape[-1]).std(0) + 1e-3, device=DEV)
        self.X = torch.from_numpy(X).to(DEV, torch.float16)
        groups = {"convo": (tag == -3) & clean, "near": tag == 2000 + k,
                  "other": (tag >= 0) & (tag < 1000) & (tag != k) & clean, "partial": (tag == -2) & clean,
                  "amb": (tag == -1) & clean}
        self.pos = torch.tensor(np.nonzero(tag == k)[0], device=DEV)
        self.neg = {g: torch.tensor(np.nonzero(m)[0], device=DEV) for g, m in groups.items() if m.any()}
        self.all_neg = torch.cat(list(self.neg.values()))
        self.streams = {}
        for s in STREAMS:
            d = E.load(s)
            Xs = FV.to_log2(np.asarray(store.load(p / s)["X"]), v)
            fsamp = round(self.fs * 16000)
            tags = [E.frame_tags(d, c, Xs.shape[1], fsamp) for c in range(Xs.shape[0])]
            self.streams[s] = (torch.from_numpy(Xs).to(DEV, torch.float16), tags)
        self.pos_last = max(2, round(0.224 / self.fs))
        self.warm = round(0.48 / self.fs)

    def norm(self, x):
        return (x.float() - self.mu) / self.sd

    def augment(self, x):
        n, _, b = x.shape
        lin = torch.linspace(-1, 1, b, device=DEV)
        x = x + (torch.rand(n, 1, 1, device=DEV) * 2 - 1) * 0.5 * lin + torch.randn(n, 1, b, device=DEV) * 0.25
        m = torch.rand(n, device=DEV) < 0.3
        bands = torch.randint(0, b, (n,), device=DEV)
        mask = F.one_hot(bands, b).bool() & m[:, None]
        return torch.where(mask[:, None, :], x.mean(dim=(1, 2), keepdim=True), x)

    def batch(self, bs, hard):
        n_pos = bs // 2
        n_neg = bs - n_pos
        n_hard = int(0.3 * n_neg) if hard is not None else 0
        idx = [self.pos[torch.randint(len(self.pos), (n_pos,), device=DEV)]]
        rest = n_neg - n_hard
        ws = {g: NEG_WEIGHTS[g] for g in self.neg}
        tot = sum(ws.values())
        for g, ids in self.neg.items():
            idx.append(ids[torch.randint(len(ids), (max(1, int(round(ws[g] / tot * rest))),), device=DEV)])
        if n_hard:
            idx.append(hard[torch.randint(len(hard), (n_hard,), device=DEV)])
        idx = torch.cat(idx)
        x = (self.augment(self.X[idx].float()) - self.mu) / self.sd
        return x, n_pos

    @torch.no_grad()
    def stream_scores(self, model, s, half):
        X, tags = self.streams[s]
        ids = list(E.chunk_ids(E.load(s), half))
        sc = []
        for i in range(0, len(ids), 32):
            sl = ids[i : i + 32]
            sc.extend(model(self.norm(X[sl])).float().cpu().numpy())
        return sc, [tags[c] for c in ids]


def loss_fn(l, n_pos, pos_last, warm):
    lp, ln = l[:n_pos], l[n_pos:, warm:]
    return (F.binary_cross_entropy_with_logits(lp[:, -pos_last:].max(1).values, torch.ones(len(lp), device=DEV))
            + F.binary_cross_entropy_with_logits(ln.max(1).values, torch.zeros(len(ln), device=DEV))
            + F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln)))


def evaluate(model, D, half, real=False):
    model.eval()
    sc, tg = D.stream_scores(model, "pos_telescope", half)
    pos = E.region_max(sc, tg)
    negs = [E.Neg(*D.stream_scores(model, "sim_convo", half), D.fs),
            E.Neg(*D.stream_scores(model, "sim_hard", half), D.fs, own=(D.k,))]
    reals = [E.Neg(*D.stream_scores(model, "real_ami", "all"), D.fs)] if real else []
    model.train()
    return E.report(pos, negs, reals)


def run(D, seed, steps=4000, width=64, lr=2e-3, bs=512, val_every=500):
    torch.manual_seed(seed)
    stride = max(1, round(0.032 / D.fs))
    model = TCN(D.X.shape[-1], stride, width).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * s / steps))))
    best, curve, hard, t0 = (9.0, None, 0), [], None, time.time()
    for step in range(1, steps + 1):
        if step % 1000 == 0:
            with torch.no_grad():
                model.eval()
                cand = D.all_neg[torch.randint(len(D.all_neg), (16384,), device=DEV)]
                sc = torch.cat([model(D.norm(D.X[cand[i : i + 2048]]))[:, D.warm :].max(1).values
                                for i in range(0, len(cand), 2048)])
                hard = cand[torch.topk(sc, 2048).indices]
                model.train()
        x, n_pos = D.batch(bs, hard)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = model(x).float()
        loss = loss_fn(l, n_pos, D.pos_last, D.warm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % val_every == 0:
            r = evaluate(model, D, "val")
            score = float(np.mean([r["frr@1.0"], r["frr@2.0"], r["frr@5.0"]]))
            curve.append({"step": step, "val_score": score, "loss": loss.item(), "time_s": round(time.time() - t0)})
            if score < best[0]:
                best = (score, {k: t.detach().clone() for k, t in model.state_dict().items()}, step)
    model.load_state_dict(best[1])
    res = {"best_step": best[2], "val_score": best[0], "curve": curve, "params": n_params(model),
           "stride": stride, "train_s": round(time.time() - t0)}
    res.update(evaluate(model, D, "test", real=True))
    return model, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--features-only", action="store_true")
    ap.add_argument("--wait-features", action="store_true", help="another process computes features")
    args = ap.parse_args()
    out = PROBE / "results" / "frontend"
    out.mkdir(parents=True, exist_ok=True)
    for name in args.variants.split(","):
        v = FV.variant(**VARIANTS[name]) if name in VARIANTS else FV.variant(**json.loads(name.split("=", 1)[1]))
        name = name.split("=", 1)[0]
        if args.wait_features:
            while not store.exists(feat_path(name) / "real_ami"):
                time.sleep(20)
        build_features(name, v)
        if args.features_only:
            continue
        D = None
        for seed in map(int, args.seeds.split(",")):
            f = out / f"{name}{args.tag}_s{seed}.json"
            if f.exists():
                continue
            D = D or Data(name, v)
            _, res = run(D, seed, args.steps, args.width)
            hw = {"variant": name, "fe": v, "seed": seed, "width": args.width, "ff": FV.flipflops(v),
                  "fe_tiles": None if FV.flipflops(v) is None else round(FV.flipflops(v) / 320, 2),
                  "cycles_per_sample": FV.cycles_per_sample(v) if v["kind"] == "chip" else None,
                  "cycle_budget": FV.cycle_budget(v), "frame_ms": 1000 * FV.frame_s(v)}
            res = {**hw, **res}
            f.write_text(json.dumps(res, indent=1))
            log(f"{name}{args.tag} s{seed}: frr@1 {res['frr@1.0']:.3f} @2 {res['frr@2.0']:.3f} @5 {res['frr@5.0']:.3f} "
                f"real_fa@1 {res['real_fa@1.0']} ff {hw['ff']} ({res['train_s']}s, best {res['best_step']})")
        del D
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
