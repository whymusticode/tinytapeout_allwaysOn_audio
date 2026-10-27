"""Probe P1: off-the-shelf wake-word models, as published, on our probe streams.

  python model/probe_pretrained.py          # -> data/probe/results/pretrained.json

openWakeWord 0.4.0 (pip): frozen Google speech embedding + per-word head, one score per 80 ms.
microWakeWord v2 (esphome/micro-wake-word-models): streaming int8 TFLite, 40 micro_frontend
features per 10 ms, one probability per 30 ms; ESPHome triggers when the mean of the last
`sliding_window_size` probabilities exceeds `probability_cutoff`, so that mean is the score here.

Positives: pos_<word> (our simulator, held-out voices, test-mode rooms/distances/ambience).
Negatives: sim_convo + sim_hard (simulated), real_ami (real meetings). Whole streams (val+test
halves): nothing here is trained or tuned.
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import probe_eval as E

ROOT = Path(__file__).resolve().parent.parent
OWW = ROOT / ".venv/lib/python3.13/site-packages/openwakeword/resources/models"
MWW = ROOT / "data/probe/mww"
WORDS = {"alexa": ("alexa_v0.1", "alexa"), "hey_jarvis": ("hey_jarvis_v0.1", "hey_jarvis"),
         "hey_mycroft": ("hey_mycroft_v0.1", "hey_mycroft"), "okay_nabu": (None, "okay_nabu")}
NEG = ("sim_convo", "sim_hard", "real_ami")

_cache = {}


def _oww(word):
    import onnxruntime as ort
    from openwakeword.utils import AudioFeatures

    if "af" not in _cache:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        _cache["af"] = AudioFeatures(ncpu=1)
    if word not in _cache:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        _cache[word] = ort.InferenceSession(str(OWW / f"{word}.onnx"), so, providers=["CPUExecutionProvider"])
    return _cache["af"], _cache[word]


def oww_chunk(job):
    """-> {word: (scores, frame end samples)} for one 60 s chunk."""
    name, c, words = job
    s = E.load(name)
    a = np.array(s["audio"][s["chunk"][c] : s["chunk"][c + 1]])
    af, _ = _oww(words[0])
    emb = af._get_embeddings(a)  # (n_emb, 96), one per 80 ms
    win = np.lib.stride_tricks.sliding_window_view(emb, 16, axis=0).transpose(0, 2, 1).astype(np.float32)
    ends = (8 * (np.arange(len(win)) + 15) + 75) * 160 + 400
    out = {}
    for w in words:
        _, sess = _oww(w)
        inp = sess.get_inputs()[0].name
        if sess.get_inputs()[0].shape[0] == 1:
            sc = np.concatenate([sess.run(None, {inp: win[i : i + 1]})[0].ravel() for i in range(len(win))])
        else:
            sc = sess.run(None, {inp: win})[0].ravel()
        out[w] = (sc, ends)
    return name, c, out


def mww_chunk(job):
    name, c, words = job
    from ai_edge_litert.interpreter import Interpreter
    from pymicro_features import MicroFrontend

    s = E.load(name)
    a = np.array(s["audio"][s["chunk"][c] : s["chunk"][c + 1]])
    fe = MicroFrontend()
    feats, i = [], 0
    raw = a.tobytes()
    while i + 320 <= len(raw):  # 10 ms of int16
        r = fe.process_samples(raw[i : i + 320])
        i += 2 * r.samples_read if r.samples_read else 320
        if r.features:
            feats.append(r.features)
    feats = np.array(feats, np.float32)
    out = {}
    for w in words:
        cfg = json.loads((MWW / f"{w}.json").read_text())["micro"]
        it = Interpreter(model_path=str(MWW / f"{w}.tflite"), num_threads=1)
        it.allocate_tensors()
        di, do = it.get_input_details()[0], it.get_output_details()[0]
        sc_in, zp_in = di["quantization"]
        sc_out, zp_out = do["quantization"]
        n = len(feats) // 3
        probs = np.zeros(n, np.float32)
        q = np.clip(np.round(feats / sc_in + zp_in), -128, 127).astype(np.int8)
        for j in range(n):
            it.set_tensor(di["index"], q[None, 3 * j : 3 * j + 3])
            it.invoke()
            probs[j] = (it.get_tensor(do["index"]).astype(np.float32).ravel()[0] - zp_out) * sc_out
        k = cfg["sliding_window_size"]
        sm = np.convolve(probs, np.ones(k) / k)[: len(probs)]
        ends = (np.arange(n) * 3 + 3) * 160
        out[w] = (sm, ends, cfg["probability_cutoff"])
    return name, c, out


def collect(fn, streams, words, n_proc):
    jobs = [(n, c, words) for n in streams for c in E.chunk_ids(E.load(n), "all")]
    res = {}
    with Pool(n_proc) as pool:
        for i, (n, c, out) in enumerate(pool.imap_unordered(fn, jobs, chunksize=2)):
            for w, v in out.items():
                res.setdefault(w, {}).setdefault(n, {})[c] = v
            if i % 200 == 0:
                print(f"{fn.__name__}: {i}/{len(jobs)}", file=sys.stderr, flush=True)
    return res


def evaluate(res_w, word_stream, default_thr=None):
    def tagged(n):
        s = E.load(n)
        sc, tg = [], []
        for c in sorted(res_w[n]):
            v = res_w[n][c]
            sc.append(v[0])
            ends = v[1]
            t10 = s["tag"][s["chunk"][c] // E.TAG_RES : s["chunk"][c + 1] // E.TAG_RES]
            tg.append(t10[np.minimum((ends - 1) // E.TAG_RES, len(t10) - 1)])
        return sc, tg

    frame_s = float(np.median(np.diff(next(iter(res_w["sim_convo"].values()))[1]))) / 16000
    sc, tg = tagged(word_stream)
    pos = E.region_max(sc, tg)
    sim = [E.Neg(*tagged(n), frame_s) for n in ("sim_convo", "sim_hard")]
    real = [E.Neg(*tagged("real_ami"), frame_s)]
    r = E.report(pos, sim, real)
    r["pos_max"] = [round(float(x), 5) for x in pos]  # per keyword region, in pos_*_meta.json order
    r["thr_sim"] = {fa: E.thr_for(sim, fa) for fa in E.FA_TARGETS}
    r["thr_real"] = {fa: E.thr_for(real, fa) for fa in E.FA_TARGETS}
    if default_thr is not None:
        r["default_thr"] = default_thr
        r["frr@default"] = float(np.mean(pos < default_thr))
        r["sim_fa@default"] = round(E.fa_rate(sim, default_thr), 2)
        r["real_fa@default"] = round(E.fa_rate(real, default_thr), 2)
    return r


def fetch_mww():
    """microWakeWord v2 models from github.com/esphome/micro-wake-word-models (once)."""
    import urllib.request

    MWW.mkdir(parents=True, exist_ok=True)
    for _, w in WORDS.values():
        for ext in ("json", "tflite"):
            f = MWW / f"{w}.{ext}"
            if not f.exists():
                urllib.request.urlretrieve(
                    f"https://raw.githubusercontent.com/esphome/micro-wake-word-models/main/models/v2/{w}.{ext}", f)


def main():
    fetch_mww()
    out = ROOT / "data/probe/results"
    out.mkdir(parents=True, exist_ok=True)
    pos = [f"pos_{w}" for w in WORDS]
    results = {}
    oww_words = [v[0] for v in WORDS.values() if v[0]]
    mww_words = [v[1] for v in WORDS.values()]
    r_mww = collect(mww_chunk, pos + list(NEG), mww_words, 12)
    for w, (_, m) in WORDS.items():
        cut = next(iter(r_mww[m]["sim_convo"].values()))[2]
        results[f"microWakeWord/{w}"] = evaluate(r_mww[m], f"pos_{w}", cut)
        print(w, "mww", results[f"microWakeWord/{w}"], file=sys.stderr, flush=True)
    (out / "pretrained.json").write_text(json.dumps(results, indent=1))
    r_oww = collect(oww_chunk, pos + list(NEG), oww_words, 12)
    for w, (o, _) in WORDS.items():
        if o:
            results[f"openWakeWord/{w}"] = evaluate(r_oww[o], f"pos_{w}", 0.5)
            print(w, "oww", results[f"openWakeWord/{w}"], file=sys.stderr, flush=True)
    (out / "pretrained.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
