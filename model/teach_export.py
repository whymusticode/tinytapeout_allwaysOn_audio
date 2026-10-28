"""Distillation targets from a Whisper encoder layer, for every Stage A audio set.

  python model/teach_export.py check     # truncated vs 30 s padded encoder: 66-word accuracy
  python model/teach_export.py export    # -> data/probe/distill/whisper_tiny_L4/<set>/X.npy
  python model/teach_export.py export --causal   # -> .../whisper_tiny_L4_causal/<set>/X.npy

A0 (probe_teachers.py) picked whisper_tiny layer 4: 69.6% at PCA-96 vs 61.1% for Google's
speech_embedding. Whisper pads every clip to 30 s, which makes 360k windows take hours; here the
encoder runs on the 1.6 s window alone (positional embeddings sliced to its 80 frames), ~20x
cheaper. `check` measures what that costs in word accuracy before `export` relies on it.

Targets: the 80 ms pooled steps j = 9..19 of each 1.6 s window (ends (j + 1) * 1280 samples, from
0.8 s to the window end), i.e. the 11 steps where the chip has heard >= 0.8 s, like the 11 steps
of speech_embedding. Reduced to 96 dims by a PCA fit on the training sets (float16).
The encoder is bidirectional within the window: step 9 has heard 0.8 s of future audio that the
student never will. Result: a 50k TCN taught by it reached word_acc 0.515 vs 0.523 taught by
speech_embedding, so the extra teacher accuracy came from the future audio.
--causal: step j's target is computed from the window up to its end only (one encoder pass per
step on audio[:, :(j + 1) * 1280], keep the last 80 ms bin): what a causal student can copy.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import probe_teachers as PT
import store

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "data/probe"
DEV = "cuda"
HF, LAYER, NAME = "openai/whisper-tiny", 4, "whisper_tiny_L4"
FIRST = 9  # first kept 80 ms step
TRAIN = ["windows_telescope", "multi_train", "windows_computer", "windows_marvin", "windows_destiny",
         "windows_shepherd", "windows_tourists"]
EVAL = ["multi_test", "real_val", "real_test"]


class Encoder:
    def __init__(self):
        from transformers import WhisperFeatureExtractor, WhisperModel

        fe = WhisperFeatureExtractor.from_pretrained(HF)
        self.enc = WhisperModel.from_pretrained(HF, torch_dtype=torch.float16).encoder.to(DEV).eval()
        self.mel = torch.from_numpy(fe.mel_filters).to(DEV).float()  # (201, 80)
        self.win = torch.hann_window(400, device=DEV)

    def logmel(self, x):
        """Whisper's log-mel (feature_extraction_whisper._torch_extract_fbank_features), no padding."""
        s = torch.stft(x, 400, 160, window=self.win, return_complex=True)[..., :-1].abs() ** 2
        m = torch.log10((self.mel.T @ s).clamp(min=1e-10))
        m = torch.maximum(m, m.amax((1, 2), keepdim=True) - 8.0)
        return (m + 4.0) / 4.0

    @torch.no_grad()
    def layer(self, audio, bs=512):
        """int16 (N, n_s) -> (N, n_s // 1280, 384) float16, hidden state after LAYER blocks."""
        e, out = self.enc, []
        for i in range(0, len(audio), bs):
            x = torch.from_numpy(audio[i : i + bs].astype(np.float32) / 32768).to(DEV)
            m = self.logmel(x).half()
            h = torch.nn.functional.gelu(e.conv2(torch.nn.functional.gelu(e.conv1(m)))).permute(0, 2, 1)
            h = h + e.embed_positions.weight[: h.shape[1]]
            for blk in e.layers[:LAYER]:
                h = blk(h, attention_mask=None, layer_head_mask=None)[0]
            if LAYER == len(e.layers):
                h = e.layer_norm(h)  # HF hidden_states[-1] is after the final norm
            out.append(PT.pool80(h.float(), x.shape[1]).half().cpu().numpy())
        return np.concatenate(out)


def check():
    E = Encoder()
    tr, te = store.load(PROBE / "multi_train"), store.load(PROBE / "multi_test")
    ytr, yte = np.asarray(tr["tag"]), np.asarray(te["tag"])
    t0 = time.time()
    a, b = E.layer(np.asarray(tr["audio"])), E.layer(np.asarray(te["audio"]))
    dt = time.time() - t0
    pa, pb = PT.pca(a, b)
    res = {"clips_per_s": round((len(a) + len(b)) / dt),
           "acc_full_20": PT.head_acc(a, ytr, b, yte)[0], "acc_pca96_20": PT.head_acc(pa, ytr, pb, yte)[0],
           "acc_pca96_11": PT.head_acc(pa[:, FIRST:], ytr, pb[:, FIRST:], yte)[0],
           "padded_30s": {k: json.loads((PROBE / "results/teachers.json").read_text())[f"whisper_tiny/L{LAYER}"][k]
                          for k in ("acc_full", "acc_pca96")}}
    print(json.dumps(res, indent=1))
    (PROBE / "results" / f"{NAME}_truncated_check.json").write_text(json.dumps(res, indent=1))


def causal_layer(E, a):
    return np.stack([E.layer(a[:, : (j + 1) * 1280])[:, j] for j in range(FIRST, 20)], 1)


def export(causal=False):
    E = Encoder()
    out = PROBE / "distill" / (NAME + ("_causal" if causal else ""))
    H = {}
    for s in TRAIN + EVAL:
        t0 = time.time()
        A = store.load(PROBE / s)["audio"]
        f = (lambda a: causal_layer(E, a)) if causal else (lambda a: E.layer(a)[:, FIRST:])
        H[s] = np.concatenate([f(np.asarray(A[i : i + 8192])) for i in range(0, len(A), 8192)])
        print(f"{s}: {H[s].shape} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    rng = np.random.default_rng(0)
    fit = np.concatenate([H[s][rng.choice(len(H[s]), min(len(H[s]), 6000), replace=False)] for s in TRAIN])
    D = fit.shape[-1]
    a = torch.from_numpy(fit.reshape(-1, D)).to(DEV).float()
    mu = a.mean(0)
    _, S, V = torch.linalg.svd(a - mu, full_matrices=False)
    P = V[:96].T
    var96 = float((S[:96] ** 2).sum() / (S ** 2).sum())
    for s, h in H.items():
        y = np.concatenate([((torch.from_numpy(h[i : i + 8192].reshape(-1, D)).to(DEV).float() - mu) @ P)
                            .reshape(-1, h.shape[1], 96).half().cpu().numpy() for i in range(0, len(h), 8192)])
        store.save(out / s, X=y)
    meta = {"hf": HF, "layer": LAYER, "causal": causal, "steps": f"80 ms bins {FIRST}..19 of 1.6 s windows, ends (j+1)*1280",
            "encoder_input": "1.6 s window only (truncated positions)", "pca": "96 dims fit on train sets",
            "pca_var_kept": round(var96, 4), "sets": {s: list(h.shape) for s, h in H.items()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check", "export"])
    ap.add_argument("--causal", action="store_true")
    a = ap.parse_args()
    check() if a.what == "check" else export(a.causal)
