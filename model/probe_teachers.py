"""Stage A0: pick the distillation teacher. Frozen teacher + the same small head, 66-word accuracy.

  python model/probe_teachers.py                 # all teachers -> data/probe/results/teachers.json
  python model/probe_teachers.py --only hubert   # one teacher (resumes: finished layers skipped)

The student copies the teacher, so the teacher's word information is the student's ceiling.
Each teacher runs on the 66-word clips (data/probe/multi_{train,test}: 1.6 s, word at the end,
test = held-out voices in the test mixer: distance, rooms, ambience). Hidden states of a few
layers are pooled to 80 ms steps (the speech_embedding rate, 20 steps per clip), then:
  acc_full   head on the full-width features
  acc_pca96  head on the top-96 PCA dims (fit on train): what a student with a 96-d output
             projection could copy at best (a 768-d target would cost the student ~50k params)
Head: standardize, flatten (steps x dims), dropout 0.3, 64 hidden, 66-way; AdamW 1e-3, 3000 steps,
2 seeds. Same for every teacher, including Google's speech_embedding (11 steps x 96, as distilled).

Bidirectional teachers (HuBERT, WavLM, wav2vec2, Whisper) see the whole 1.6 s clip, as they
would see a whole training window during distillation; speech_embedding is causal (0.775 s).
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

import store

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "data/probe"
OUT = PROBE / "results/teachers.json"
TEACH = PROBE / "teach"  # pooled PCA-96 features of the kept teachers
DEV = "cuda"
STEP = 1280  # 80 ms at 16 kHz

# name: (hf id, layers to probe, params, license)
TEACHERS = {
    "hubert_base": ("facebook/hubert-base-ls960", (3, 6, 9, 12), "95M", "Apache-2.0"),
    "wavlm_base_plus": ("microsoft/wavlm-base-plus", (3, 6, 9, 12), "95M", "MIT"),
    "wav2vec2_base": ("facebook/wav2vec2-base", (3, 6, 9, 12), "95M", "Apache-2.0"),
    "w2v2_phoneme": ("facebook/wav2vec2-lv-60-espeak-cv-ft", (8, 12, 16, 20, 24), "315M", "Apache-2.0"),
    "whisper_tiny": ("openai/whisper-tiny", (2, 4), "8M enc", "MIT"),
    "whisper_base": ("openai/whisper-base", (2, 4, 6), "20M enc", "MIT"),
}


def pool80(h, n_samples):
    """(B, T, D) frames covering n_samples -> (B, n_samples // STEP, D) means per 80 ms."""
    n = n_samples // STEP
    T = h.shape[1]
    edges = np.linspace(0, T, n + 1).round().astype(int)
    return torch.stack([h[:, a:b].mean(1) for a, b in zip(edges[:-1], edges[1:])], 1)


@torch.no_grad()
def hidden(name, audio, bs=48):
    """-> {layer: (N, 20, D) float16 numpy} for the probed layers."""
    hf, layers, _, _ = TEACHERS[name]
    n_s = audio.shape[1]
    out = {l: [] for l in layers}
    if name.startswith("whisper"):
        from transformers import WhisperFeatureExtractor, WhisperModel

        fe = WhisperFeatureExtractor.from_pretrained(hf)
        enc = WhisperModel.from_pretrained(hf, torch_dtype=torch.float16).encoder.to(DEV).eval()
        n_fr = n_s // 320  # encoder frames are 20 ms
        for i in range(0, len(audio), bs):
            x = audio[i : i + bs].astype(np.float32) / 32768
            f = fe(list(x), sampling_rate=16000, return_tensors="pt").input_features.to(DEV, torch.float16)
            hs = enc(f, output_hidden_states=True).hidden_states
            for l in layers:
                out[l].append(pool80(hs[l][:, :n_fr].float(), n_s).half().cpu().numpy())
    else:
        from transformers import AutoModel

        m = AutoModel.from_pretrained(hf, torch_dtype=torch.float16).to(DEV).eval()
        for i in range(0, len(audio), bs):
            x = torch.from_numpy(audio[i : i + bs].astype(np.float32) / 32768).to(DEV)
            x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-5)  # processor's normalization
            hs = m(x.half(), output_hidden_states=True).hidden_states
            for l in layers:
                out[l].append(pool80(hs[l].float(), n_s).half().cpu().numpy())
    return {l: np.concatenate(v) for l, v in out.items()}


def pca(tr, te, k=96):
    D = tr.shape[-1]
    a = torch.from_numpy(tr.reshape(-1, D)).to(DEV).float()
    mu = a.mean(0)
    _, _, V = torch.linalg.svd(a[torch.randperm(len(a), device=DEV)[:200_000]] - mu, full_matrices=False)
    P = V[:k].T

    def f(x):
        return ((torch.from_numpy(x.reshape(-1, D)).to(DEV).float() - mu) @ P).reshape(*x.shape[:-1], k).half().cpu().numpy()

    return f(tr), f(te)


def head_acc(tr, ytr, te, yte, seeds=(0, 1), steps=3000, bs=256):
    mu, sd = tr.reshape(-1, tr.shape[-1]).astype(np.float32).mean(0), tr.reshape(-1, tr.shape[-1]).astype(np.float32).std(0) + 1e-3
    Xtr = torch.from_numpy((tr - mu) / sd).to(DEV, torch.float16)
    Xte = torch.from_numpy((te - mu) / sd).to(DEV, torch.float16)
    ytr_t, yte_t = torch.from_numpy(ytr).long().to(DEV), torch.from_numpy(yte).long().to(DEV)
    n_cls = int(ytr.max()) + 1
    accs = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(tr.shape[1] * tr.shape[2], 64), nn.LayerNorm(64),
                              nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_cls)).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
        for _ in range(steps):
            i = torch.randint(len(Xtr), (bs,), device=DEV)
            x = Xtr[i].float()
            loss = F.cross_entropy(model(x + 0.05 * torch.randn_like(x)), ytr_t[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(Xte[j : j + 2048].float()).argmax(1) for j in range(0, len(Xte), 2048)])
        accs.append(float((pred == yte_t).float().mean()))
    return round(float(np.mean(accs)), 4), [round(a, 4) for a in accs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of teachers")
    args = ap.parse_args()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    tr = store.load(PROBE / "multi_train")
    te = store.load(PROBE / "multi_test")
    ytr, yte = np.asarray(tr["tag"]), np.asarray(te["tag"])

    def save():
        OUT.write_text(json.dumps(res, indent=1))

    if "speech_embedding" not in res:  # the current teacher, as distilled (11 x 96 per clip)
        a, b = (np.asarray(store.load(PROBE / "emb" / s)["X"]) for s in ("multi_train", "multi_test"))
        acc, per = head_acc(a, ytr, b, yte)
        res["speech_embedding"] = {"layer": "out", "dims": 96, "acc_full": acc, "acc_pca96": acc, "seeds": per,
                                   "hf": "openWakeWord / Google speech_embedding", "params": "~1.3M"}
        print("speech_embedding", res["speech_embedding"], file=sys.stderr, flush=True)
        save()
    names = [n for n in (args.only.split(",") if args.only else TEACHERS)]
    for name in names:
        hf, layers, params, lic = TEACHERS[name]
        todo = [l for l in layers if f"{name}/L{l}" not in res]
        if not todo:
            continue
        t0 = time.time()
        Htr = hidden(name, np.asarray(tr["audio"]))
        Hte = hidden(name, np.asarray(te["audio"]))
        t_emb = round(time.time() - t0)
        for l in todo:
            a, b = Htr[l], Hte[l]
            acc_full, per_full = head_acc(a, ytr, b, yte)
            pa, pb = pca(a, b)
            acc_p, per_p = head_acc(pa, ytr, pb, yte)
            store.save(TEACH / f"{name}_L{l}" / "multi_train", X=pa)
            store.save(TEACH / f"{name}_L{l}" / "multi_test", X=pb)
            res[f"{name}/L{l}"] = {"layer": l, "dims": int(a.shape[-1]), "acc_full": acc_full, "acc_pca96": acc_p,
                                   "seeds_full": per_full, "seeds_pca96": per_p, "hf": hf, "params": params,
                                   "license": lic, "embed_s_all_layers": t_emb}
            print(name, l, res[f"{name}/L{l}"], file=sys.stderr, flush=True)
            save()
        del Htr, Hte
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
