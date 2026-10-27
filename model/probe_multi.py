"""Information probe for front-end variants: 66-way word classification on held-out voices.

  python model/probe_multi.py --variants base,pcm10 --seeds 0,1

A front end that throws away speech information lowers this accuracy, whatever the detector.
It is far less noisy than a wake-word miss rate at 1 false alarm/h (6,600 test clips, every
word counts), so it ranks front ends cheaply; probe_frontend.py gives the wake-word view.

Data: data/probe/multi_{train,test} (probe_data.py multi): 300 train-voice clips per word in
simulator train mode, 100 held-out-voice clips per word in simulator test mode (held-out rooms
and ambience). Model: the probe TCN body (~20k params), class logits averaged over the last
~0.3 s. Results: data/probe/results/multi/<variant>_s<seed>.json
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
import store
from probe_frontend import TCN, VARIANTS, feat_path

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "data/probe"
DEV = "cuda"


def features(name, v):
    for split in ("train", "test"):
        p = feat_path(name) / f"multi_{split}"
        if not store.exists(p):
            A = store.load(PROBE / f"multi_{split}")["audio"]
            store.save(p, X=np.concatenate([FV.features(np.asarray(A[i : i + 4096]), v) for i in range(0, len(A), 4096)]))
    tr, te = (FV.to_log2(np.asarray(store.load(feat_path(name) / f"multi_{s}")["X"]), v) for s in ("train", "test"))
    ytr, yte = (np.asarray(store.load(PROBE / f"multi_{s}")["tag"]) for s in ("train", "test"))
    return tr, ytr, te, yte


class Clf(TCN):
    def __init__(self, n_in, stride, n_cls, width=64, pool_s=0.3, frame_s=0.016):
        super().__init__(n_in, stride, width)
        self.out = nn.Linear(width, n_cls)
        self.pool = max(1, round(pool_s / (frame_s * stride)))

    def forward(self, x):
        B, T, C = x.shape
        s = self.stride
        pad = (-T) % s
        x = F.pad(x, (0, 0, 0, pad)).reshape(B, (T + pad) // s, s * C)
        h = self.inp(x).transpose(1, 2)
        for d, dw, pw, bn in zip(self.dil, self.dw, self.pw, self.bn):
            h = h + F.relu(bn(pw(dw(F.pad(h, (2 * d, 0))))))
        return self.out(h[:, :, -self.pool :].mean(-1))


def run(name, v, seed, steps=2500, bs=256, lr=3e-3, width=64):
    tr, ytr, te, yte = features(name, v)
    mu, sd = tr.reshape(-1, tr.shape[-1]).mean(0), tr.reshape(-1, tr.shape[-1]).std(0) + 1e-3
    Xtr = torch.from_numpy((tr - mu) / sd).to(DEV, torch.float16)
    Xte = torch.from_numpy((te - mu) / sd).to(DEV, torch.float16)
    ytr_t, yte_t = torch.from_numpy(ytr).long().to(DEV), torch.from_numpy(yte).long().to(DEV)
    torch.manual_seed(seed)
    fs = FV.frame_s(v)
    model = Clf(tr.shape[-1], max(1, round(0.032 / fs)), int(ytr.max()) + 1, width=width, frame_s=fs).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    t0 = time.time()
    for step in range(steps):
        i = torch.randint(len(Xtr), (bs,), device=DEV)
        x = Xtr[i].float()
        n, _, b = x.shape
        x = x + torch.randn(n, 1, b, device=DEV) * 0.25 / torch.from_numpy(sd).to(DEV).float()  # per-band gain jitter
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).float(), ytr_t[i], label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(Xte[i : i + 1024].float()).argmax(1) for i in range(0, len(Xte), 1024)])
        acc = float((pred == yte_t).float().mean())
        trp = torch.cat([model(Xtr[i : i + 1024].float()).argmax(1) for i in range(0, 8192, 1024)])
        tr_acc = float((trp == ytr_t[:8192]).float().mean())
    return {"variant": name, "seed": seed, "fe": v, "width": width, "params": sum(p.numel() for p in model.parameters()), "test_acc": acc, "train_acc": tr_acc,
            "ff": FV.flipflops(v), "frame_ms": 1000 * fs, "train_s": round(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--width", type=int, default=64)
    args = ap.parse_args()
    out = PROBE / "results" / "multi"
    out.mkdir(parents=True, exist_ok=True)
    for name in args.variants.split(","):
        v = FV.variant(**VARIANTS[name]) if name in VARIANTS else FV.variant(**json.loads(name.split("=", 1)[1]))
        name = name.split("=", 1)[0]
        for seed in map(int, args.seeds.split(",")):
            f = out / f"{name}{'' if args.width == 64 else f'_w{args.width}'}_s{seed}.json"
            if f.exists():
                continue
            r = run(name, v, seed, width=args.width)
            f.write_text(json.dumps(r, indent=1))
            print(f"{name} w{args.width} s{seed}: acc {r['test_acc']:.3f} (train {r['train_acc']:.3f}) ff {r['ff']} {r['train_s']}s",
                  file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
