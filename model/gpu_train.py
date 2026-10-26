"""GPU (ROCm) trainer for the architecture x size ladder, with early stopping on holdout data.

  python model/gpu_train.py --word telescope [--arch gru,tcn] [--sizes 0,2,4] --out data/results/arch_v1

Loads the training cache and the evaluation streams onto the GPU once, then trains every
requested (architecture, width) from archs.WIDTHS in turn. One JSON + checkpoint per run.

Data splits (nothing real is ever trained on or selected on):
  train   data/cache/train (simulated, 16 mixes per keyword clip)
  val     simulator holdout, half A: even held-out TTS positives, first half of sim_convo
          chunks, even sim_hard chunks. Early stopping and checkpoint choice use ONLY this.
  test    simulator holdout, half B (the other halves) -> frr_syn, fa_sim_*
          real recordings -> frr_real (Speech Commands / MSWC in real rooms), fa_real_ami
          (86 h of meetings), fa_real_read. Thresholds as in train.py.
  monitor every validation also logs, never used for any decision: real-speaker FRR and
          real-meeting FA/h (first 20 h of AMI) at the val threshold, so the curves show
          whether simulator validation tracks reality during training.

Training: AdamW, warm-up + cosine, bf16 autocast (fp32 for the cuDNN/MIOpen GRU), batch 512
(1/4 positives), GPU-side augmentation (tilt, jitter, band masks) and hard-negative mining.
Validation every VAL_EVERY steps; stop after PATIENCE validations without improvement in
val_score = mean FRR at 0.5 / 1 / 2 FA/h (sim val half).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import archs
import store
from synth_tts import load_candidates, slug
from train import CACHE, POS_LAST, REFRACTORY, WARMUP, count_fa, has_word, norm_stats

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda"
BATCH, POS_FRAC = 512, 0.25
LR, WD, WARM_STEPS, MAX_STEPS = 2e-3, 1e-2, 300, 8000
VAL_EVERY, PATIENCE = 250, 8
MINE_EVERY, MINE_POOL, MINE_KEEP, MINE_FRAC = 500, 32768, 4096, 0.3
AUG_TILT, AUG_JITTER, AUG_P_MASK = 0.5, 0.25, 0.3
VAL_FAS = (0.5, 1.0, 2.0)
FA_TARGETS = (0.5, 1.0, 2.0)
MON_AMI_CHUNKS = 300  # ~5 h of AMI for the per-validation real monitor
FRAME_S = archs.FRAME_S
NEG_WEIGHTS = {"convo": 0.35, "near": 0.10, "other": 0.25, "partial": 0.10, "amb": 0.20}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --- data ------------------------------------------------------------------------------

class Stream:
    """A set of 60 s chunks on the GPU, grouped by length for batched evaluation."""

    def __init__(self, name, own, chunk_ids=None):
        s = store.load(CACHE / f"stream_{name}")
        chunk, tag = np.array(s["chunk"]), np.array(s["tag"])
        ids = np.arange(len(chunk) - 1) if chunk_ids is None else np.asarray(chunk_ids)
        parts = [(chunk[i], chunk[i + 1]) for i in ids]
        idx = np.concatenate([np.arange(a, b) for a, b in parts])
        self.X = torch.from_numpy(np.asarray(s["X"])[idx]).to(DEV)
        self.ignore = np.isin(tag[idx], list(own))
        lens = np.array([b - a for a, b in parts])
        self.chunk = np.concatenate([[0], np.cumsum(lens)])
        self.hours = len(idx) * FRAME_S / 3600
        self.groups = {}
        for L in np.unique(lens):
            self.groups[int(L)] = torch.tensor(self.chunk[:-1][lens == L], device=DEV)

    @torch.no_grad()
    def logits(self, model, norm, max_elems=2e8):
        out = torch.full((len(self.X),), -20.0, device=DEV)
        for L, starts in self.groups.items():
            bs = max(1, int(max_elems // (L * 4 * model.width)))
            ar = torch.arange(L, device=DEV)
            for i in range(0, len(starts), bs):
                idx = starts[i : i + bs, None] + ar  # (b, L)
                with autocast(model):
                    l = model(norm(self.X[idx])).float()
                l[:, :WARMUP] = -20.0
                out[idx] = l
        return out.cpu().numpy()


def autocast(model):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=model.arch != "gru")


def fa_rate(S, thr):
    return sum(count_fa(l, s.ignore, s.chunk, thr) for s, l in S) / sum(s.hours for s, _ in S)


def thr_for(S, target):
    lo, hi = -20.0, 40.0
    for _ in range(36):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fa_rate(S, mid) > target else (lo, mid)
    return hi


class Data:
    def __init__(self, k):
        t0 = time.time()
        self.k = k
        own = {k, 1000 + k}
        tr = store.load(CACHE / "train")
        tag, contains = np.array(tr["tag"]), np.array(tr["contains"])
        mu, sd = norm_stats(tr["X"], tag)
        self.mu, self.sd = torch.tensor(mu, device=DEV), torch.tensor(sd, device=DEV)
        self.X = torch.from_numpy(np.asarray(tr["X"])).to(DEV)  # (N, 100, 16) uint8, ~4 GB
        clean = ~has_word(contains, k)
        groups = {
            "convo": (tag == -3) & clean,
            "near": tag == 2000 + k,
            "other": (tag >= 0) & (tag != k) & (tag != 2000 + k) & clean,
            "partial": (tag == -2) & clean,
            "amb": (tag == -1) & clean,
        }
        self.pos = torch.tensor(np.nonzero(tag == k)[0], device=DEV)
        self.neg = {g: torch.tensor(np.nonzero(m)[0], device=DEV) for g, m in groups.items() if m.any()}
        self.all_neg = torch.cat(list(self.neg.values()))
        # held-out positives: simulator (split val/test by parity) and real recordings
        te = store.load(CACHE / "test")
        ttag = np.array(te["tag"])
        tX = np.asarray(te["X"])
        syn = np.nonzero(ttag == k)[0]
        self.pos_val = torch.from_numpy(tX[syn[0::2]]).to(DEV)
        self.pos_test = torch.from_numpy(tX[syn[1::2]]).to(DEV)
        real = np.nonzero(ttag == 1000 + k)[0]
        self.pos_real = torch.from_numpy(tX[real]).to(DEV) if len(real) else None
        # streams
        nc = len(np.array(store.load(CACHE / "stream_sim_convo")["chunk"])) - 1
        nh = len(np.array(store.load(CACHE / "stream_sim_hard")["chunk"])) - 1
        self.val_streams = [Stream("sim_convo", own, range(nc // 2)), Stream("sim_hard", own, range(0, nh, 2))]
        self.test_sim = {"sim_convo": Stream("sim_convo", own, range(nc // 2, nc)),
                         "sim_hard": Stream("sim_hard", own, range(1, nh, 2))}
        self.mon_ami = Stream("real_ami", own, range(MON_AMI_CHUNKS))
        self.test_real = {"real_ami": None, "real_read": None}  # loaded lazily (large)
        self.own = own
        # fixed negative windows from the val streams for a smooth val loss
        g = torch.Generator(device="cpu").manual_seed(0)
        vX = torch.cat([s.X for s in self.val_streams])
        vign = torch.from_numpy(np.concatenate([s.ignore for s in self.val_streams]))
        starts = torch.randint(0, len(vX) - 100, (4096,), generator=g)
        keep = ~torch.stack([vign[s : s + 100].any() for s in starts])
        self.val_negwin = vX[(starts[keep, None] + torch.arange(100)).to(DEV)]
        log(f"data for word {k} on GPU in {time.time() - t0:.0f}s: train {tuple(self.X.shape)}, "
            f"pos {len(self.pos)}, val pos {len(self.pos_val)}, test pos {len(self.pos_test)}, "
            f"real pos {0 if self.pos_real is None else len(self.pos_real)}, "
            f"val streams {sum(s.hours for s in self.val_streams):.1f} h")

    def norm(self, x):
        return (x.float() / 8 - self.mu) / self.sd

    def augment(self, x):
        """x: float (n, T, 16) log2 units."""
        n, _, b = x.shape
        lin = torch.linspace(-1, 1, b, device=DEV)
        x = x + (torch.rand(n, 1, 1, device=DEV) * 2 - 1) * AUG_TILT * lin + torch.randn(n, 1, b, device=DEV) * AUG_JITTER
        m = torch.rand(n, device=DEV) < AUG_P_MASK
        if m.any():
            bands = torch.randint(0, b, (n, 2), device=DEV)
            mask = torch.zeros(n, b, dtype=torch.bool, device=DEV)
            mask.scatter_(1, bands[:, :1], True)
            mask.scatter_(1, bands[:, 1:], torch.rand(n, 1, device=DEV) < 0.5)
            mask &= m[:, None]
            x = torch.where(mask[:, None, :], x.mean(dim=(1, 2), keepdim=True), x)
        return x

    def batch(self, hard):
        n_pos = int(BATCH * POS_FRAC)
        n_neg = BATCH - n_pos
        n_hard = int(MINE_FRAC * n_neg) if hard is not None else 0
        idx = [self.pos[torch.randint(len(self.pos), (n_pos,), device=DEV)]]
        rest = n_neg - n_hard
        for g, ids in self.neg.items():
            idx.append(ids[torch.randint(len(ids), (int(round(NEG_WEIGHTS[g] * rest)),), device=DEV)])
        if n_hard:
            idx.append(hard[torch.randint(len(hard), (n_hard,), device=DEV)])
        idx = torch.cat(idx)
        x = (self.augment(self.X[idx].float() / 8) - self.mu) / self.sd
        return x, n_pos


# --- training ----------------------------------------------------------------------------

def loss_fn(lp, ln):
    """positive windows: fire somewhere in the last POS_LAST frames; negatives: never after warm-up."""
    loss_p = F.binary_cross_entropy_with_logits(lp[:, -POS_LAST:].max(1).values, torch.ones(len(lp), device=DEV))
    ln = ln[:, WARMUP:]
    loss_n = F.binary_cross_entropy_with_logits(ln.max(1).values, torch.zeros(len(ln), device=DEV)) + \
        F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))
    return loss_p + loss_n


@torch.no_grad()
def window_max(model, X, norm, last=None, bs=4096):
    out = []
    for i in range(0, len(X), bs):
        with autocast(model):
            l = model(norm(X[i : i + bs])).float()
        out.append(l[:, -last:].max(1).values if last else l[:, WARMUP:].max(1).values)
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def validate(model, D):
    model.eval()
    S = [(s, s.logits(model, D.norm)) for s in D.val_streams]
    pos = window_max(model, D.pos_val, D.norm, last=POS_LAST)
    res = {}
    for fa in VAL_FAS:
        res[f"val_frr@{fa}"] = float(np.mean(pos <= thr_for(S, fa)))
    res["val_score"] = float(np.mean([res[f"val_frr@{fa}"] for fa in VAL_FAS]))
    with autocast(model):
        lp = model(D.norm(D.pos_val)).float()
        ln = model(D.norm(D.val_negwin)).float()
    res["val_loss"] = float(loss_fn(lp, ln))
    thr = thr_for(S, 1.0)  # monitor only
    if D.pos_real is not None:
        res["mon_frr_real@1.0"] = float(np.mean(window_max(model, D.pos_real, D.norm) <= thr))
    res["mon_fa_ami@1.0"] = round(fa_rate([(D.mon_ami, D.mon_ami.logits(model, D.norm))], thr), 3)
    model.train()
    return res


@torch.no_grad()
def test(model, D):
    model.eval()
    for n in D.test_real:
        if D.test_real[n] is None:
            D.test_real[n] = Stream(n, D.own)
    S = {n: (s, s.logits(model, D.norm)) for n, s in {**D.test_sim, **D.test_real}.items()}
    sim = [S["sim_convo"], S["sim_hard"]]
    real = [S["real_ami"], S["real_read"]]
    pos_syn = window_max(model, D.pos_test, D.norm, last=POS_LAST)
    pos_real = window_max(model, D.pos_real, D.norm) if D.pos_real is not None else None
    res = {"stream_hours": {n: round(s.hours, 2) for n, (s, _) in S.items()},
           "n_pos_syn": len(pos_syn), "n_pos_real": 0 if pos_real is None else len(pos_real)}
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
    model.train()
    return res


def run(D, arch, width, seed=0):
    torch.manual_seed(seed)
    model = archs.build(arch, width).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARM_STEPS) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, MAX_STEPS) / MAX_STEPS))))
    hard, curve, best, bad = None, [], (np.inf, np.inf, 0, None), 0
    tl, t0 = [], time.time()
    for step in range(1, MAX_STEPS + 1):
        if step % MINE_EVERY == 0:
            with torch.no_grad():
                model.eval()
                cand = D.all_neg[torch.randint(len(D.all_neg), (MINE_POOL,), device=DEV)]
                sc = torch.cat([_mine_scores(model, D, cand[i : i + 4096]) for i in range(0, MINE_POOL, 4096)])
                hard = cand[torch.topk(sc, MINE_KEEP).indices]
                model.train()
        x, n_pos = D.batch(hard)
        with autocast(model):
            l = model(x).float()
        loss = loss_fn(l[:n_pos], l[n_pos:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tl.append(loss.detach())
        if step % VAL_EVERY == 0:
            v = validate(model, D)
            rec = {"step": step, "train_loss": float(torch.stack(tl).mean()), "lr": sched.get_last_lr()[0],
                   "time_s": round(time.time() - t0, 1), **v}
            tl = []
            curve.append(rec)
            if (v["val_score"], v["val_loss"]) < best[:2]:
                best, bad = (v["val_score"], v["val_loss"], step, {k: t.detach().clone() for k, t in model.state_dict().items()}), 0
            else:
                bad += 1
            log(f"  {arch}:{width} step {step} train {rec['train_loss']:.3f} val_loss {v['val_loss']:.3f} "
                f"val_score {v['val_score']:.3f} real {v.get('mon_frr_real@1.0', float('nan')):.3f} "
                f"ami {v['mon_fa_ami@1.0']:.2f}/h ({rec['time_s']:.0f}s){' *' if bad == 0 else ''}")
            if bad >= PATIENCE:
                break
    model.load_state_dict(best[3])
    return model, {"best_step": best[2], "stopped_step": step, "train_time_s": round(time.time() - t0, 1), "curve": curve}


def _mine_scores(model, D, idx):
    with autocast(model):
        return model(D.norm(D.X[idx])).float()[:, WARMUP:].max(1).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", required=True)
    ap.add_argument("--arch", default=",".join(archs.WIDTHS))
    ap.add_argument("--sizes", default="0,1,2,3,4", help="indices into archs.WIDTHS[arch]")
    ap.add_argument("--out", default="data/results/arch_v1")
    ap.add_argument("--max-steps", type=int)
    args = ap.parse_args()
    global MAX_STEPS
    MAX_STEPS = args.max_steps or MAX_STEPS
    cands = load_candidates()
    k = [slug(c["phrase"]) for c in cands].index(args.word)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    D = Data(k)
    for arch in args.arch.split(","):
        for si in map(int, args.sizes.split(",")):
            width = archs.WIDTHS[arch][si]
            name = f"{args.word}_{arch}_{si}"
            if (out / f"{name}.json").exists():
                log(f"skip {name} (done)")
                continue
            log(f"== {name}: {arch} width {width}")
            model, hist = run(D, arch, width)
            res = {"word": cands[k]["phrase"], "slug": args.word, "group": cands[k]["group"], "arch": arch,
                   "size_index": si, "width": width, **archs.cost(model), **hist, **test(model, D)}
            res["curve"] = res.pop("curve")
            (out / f"{name}.json").write_text(json.dumps(res, indent=1))
            torch.save({"arch": arch, "width": width, "state": model.state_dict(),
                        "mu": D.mu.cpu().numpy(), "sd": D.sd.cpu().numpy()}, out / f"{name}.pt")
            log(f"   -> best step {res['best_step']}, frr_syn@1 {res['frr_syn@1.0']:.3f}, "
                f"frr_real@1 {res.get('frr_real@1.0', float('nan')):.3f}, ami {res['fa_real_ami@1.0']}/h, "
                f"frr_real@real1 {res.get('frr_real@real1', float('nan')):.3f}, {res['train_time_s']:.0f}s")


if __name__ == "__main__":
    main()
