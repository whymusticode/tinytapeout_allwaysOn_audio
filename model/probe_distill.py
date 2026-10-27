"""Probe P6: distil the pretrained speech embedding into a chip-sized student on chip features.

  python model/probe_distill.py --width 64 --fe base

Teacher: Google's speech_embedding (probe_embed.py), 96-d every 80 ms, computed on the raw
audio of the simulated training windows. Student: causal TCN on the chip's features (variant
`--fe`, cached by probe_frontend.py), trained to output the teacher's embedding at the same
instants (cosine + MSE on standardised embeddings). Simulated audio only: the student never
sees real speech, only the teacher's outputs on simulated mixes.

Then the student's embeddings replace the teacher's in the P5 pipeline (same MLP head, same
streams, same metric), so the result reads directly against P5 (teacher, ~330k+ params,
mel front end) and against the TCN trained from scratch on the same features (probe F).
Results: data/probe/results/embed_telescope_student_<fe>_w<width>_s0.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import frontend_var as FV
import probe_embed as PE
import probe_eval as E
import store
from probe_frontend import VARIANTS, feat_path

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "data/probe"
DEV = "cuda"


def emb_end(j):
    """Last 16 kHz sample of teacher embedding j (from the start of a window or chunk)."""
    return (8 * j + 75) * 160 + 400


class Student(nn.Module):
    """Frames stacked by `stride`, then causal depthwise-separable conv blocks (kernel 3,
    dilations `dil`), residual, BN; linear to 96. Output per step."""

    def __init__(self, n_in, stride, width, dil=(1, 2, 4, 8, 1, 2, 4, 8)):
        super().__init__()
        self.stride, self.dil = stride, dil
        self.inp = nn.Linear(n_in * stride, width)
        self.dw = nn.ModuleList([nn.Conv1d(width, width, 3, dilation=d, groups=width) for d in dil])
        self.pw = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in dil])
        self.bn = nn.ModuleList([nn.BatchNorm1d(width) for _ in dil])
        self.out = nn.Linear(width, 96)

    def forward(self, x):  # (B, T, C) -> (B, T // stride, 96); step i ends at frame (i+1)*stride-1
        B, T, C = x.shape
        s = self.stride
        x = x[:, : T - T % s].reshape(B, T // s, s * C)
        h = self.inp(x).transpose(1, 2)
        for d, dw, pw, bn in zip(self.dil, self.dw, self.pw, self.bn):
            h = h + F.relu(bn(pw(dw(F.pad(h, (2 * d, 0))))))
        return self.out(h.transpose(1, 2))


def step_index(n_emb, frame_samples, stride):
    """Student step whose last frame contains each teacher embedding's last sample."""
    ends = emb_end(np.arange(n_emb))
    return (ends // frame_samples) // stride  # frame containing the end sample -> step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fe", default="base")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", default="multi_train,windows_computer", help="more distillation window sets")
    ap.add_argument("--tag", default="")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    v = FV.variant(**VARIANTS[args.fe])
    fsamp = int(round(FV.frame_s(v) * 16000))
    stride = max(1, round(0.032 / FV.frame_s(v)))
    # (audio set, teacher embedding set, feature cache name)
    sets = [("windows_telescope", "windows", "windows")] + [(x, x, x) for x in args.extra.split(",") if x]
    feats, embs = [], []
    for src, e, fname in sets + [("multi_test", "multi_test", "multi_test")]:
        if not store.exists(PE.EMB / e):
            print(f"skip {src}: no teacher embeddings", file=sys.stderr)
            continue
        p = feat_path(args.fe) / fname
        if not store.exists(p):
            A = store.load(PROBE / src)["audio"]
            store.save(p, X=np.concatenate([FV.features(np.asarray(A[i : i + 4096]), v) for i in range(0, len(A), 4096)]))
        feats.append(FV.to_log2(np.asarray(store.load(p)["X"]), v))
        embs.append(np.asarray(store.load(PE.EMB / e)["X"]).astype(np.float32))
    X, T = np.concatenate(feats[:-1]), np.concatenate(embs[:-1])
    Xv, Tv = feats[-1], embs[-1]  # held-out voices, rooms, ambience: the fidelity measure
    n_tel = len(feats[0])
    mu, sd = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0) + 1e-3
    emu, esd = T.reshape(-1, 96).mean(0), T.reshape(-1, 96).std(0) + 1e-3
    Xg = torch.from_numpy(np.concatenate([X, Xv])).sub_(torch.from_numpy(mu)).div_(torch.from_numpy(sd)).to(DEV, torch.float16)
    Tg = torch.from_numpy((np.concatenate([T, Tv]) - emu) / esd).to(DEV)
    del X, T
    idx = torch.from_numpy(step_index(Tg.shape[1], fsamp, stride)).to(DEV)
    n = len(Xg)
    tr = torch.arange(n - len(Xv), device=DEV)
    va = torch.arange(n - len(Xv), n, device=DEV)
    model = Student(Xg.shape[-1], stride, args.width).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / args.steps)))

    def loss_of(i):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(Xg[i].float()).float()[:, idx]
        t = Tg[i]
        return F.mse_loss(y, t) + (1 - F.cosine_similarity(y, t, dim=-1)).mean(), F.cosine_similarity(y, t, dim=-1).mean()

    t0, best, curve_d = time.time(), (9.0, None), []
    for step in range(1, args.steps + 1):
        i = tr[torch.randint(len(tr), (args.batch,), device=DEV)]
        loss, _ = loss_of(i)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                vl, vc = zip(*[loss_of(va[j : j + 1024]) for j in range(0, len(va), 1024)])
            model.train()
            vl, vc = float(torch.stack(vl).mean()), float(torch.stack(vc).mean())
            print(f"step {step} val loss {vl:.3f} cos {vc:.3f} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
            curve_d.append({"step": step, "val_loss": vl, "val_cos": vc})
            if vl < best[0]:
                best = (vl, {a: b.clone() for a, b in model.state_dict().items()}, vc)
    model.load_state_dict(best[1])
    model.eval()
    # student embeddings (de-standardised to the teacher's scale) for windows and streams
    name = f"{args.fe}_w{args.width}{args.tag}"
    out = PROBE / f"emb_student_{name}"
    emu_t, esd_t = torch.from_numpy(emu).to(DEV), torch.from_numpy(esd).to(DEV)
    with torch.no_grad():
        W = torch.cat([model(Xg[j : min(n_tel, j + 2048)].float())[:, idx] for j in range(0, n_tel, 2048)]) * esd_t + emu_t
        store.save(out / "windows", X=W.half().cpu().numpy())
        for s in PE.STREAMS:
            Xs = FV.to_log2(np.asarray(store.load(feat_path(args.fe) / s)["X"]), v)
            Xs = torch.from_numpy((Xs - mu) / sd).to(DEV, torch.float16)
            n_emb = store.load(PE.EMB / s)["X"].shape[1]
            sidx = torch.from_numpy(step_index(n_emb, fsamp, stride)).to(DEV)
            Y = torch.cat([model(Xs[j : j + 16].float())[:, sidx] for j in range(0, len(Xs), 16)]) * esd_t + emu_t
            store.save(out / s, X=Y.half().cpu().numpy())
    print(f"student {name}: {n_par} params, held-out cos {best[2]:.3f}", file=sys.stderr, flush=True)
    PE.train(args.seed, steps=3000, lr=3e-4, hidden=64, drop=0.3, wd=0.1, val_every=100,
             out_tag=f"_student_{name}", emb_dir=out)
    f = PROBE / "results" / f"embed_telescope_student_{name}_s{args.seed}.json"
    r = json.loads(f.read_text())
    r.update({"student_params": n_par, "student_val_cos": best[2], "fe": args.fe, "width": args.width,
              "distill_sets": [x[0] for x in sets], "distill_windows": int(len(tr)), "curve_distill": curve_d})
    f.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
