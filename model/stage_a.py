"""Stage A runner: distil the teacher into a chip-sized student. Resumable; one ledger line per run.

  python model/stage_a.py add research/stage_a/a1_frontend.jsonl   # queue configs (deduplicated)
  python model/stage_a.py work                                     # run the queue; Ctrl-C / SIGTERM saves and exits
  python model/stage_a.py status

The GPU is shared, so runs stop and resume: a checkpoint (model, optimizer, schedule, RNG, curve,
best weights) is written every EVAL_EVERY steps and on SIGTERM/SIGINT; `work` resumes unfinished
runs first. A hard kill loses at most EVAL_EVERY steps.

Files
  research/stage_a/queue.jsonl    configs to run (committed; `add` appends)
  research/stage_a/ledger.jsonl   one line per finished or pruned run: config, size, area, metrics
                                  (committed; research/07_stage_a reads only this)
  data/stage_a/runs/<id>/         ckpt.pt, curve.jsonl (gitignored)

Config (search_space.py names; missing keys take DEFAULTS): name, group, size, family + its
knobs under "knobs", bands/order/hop_ms (front end, lean5 bit widths), stride (16 ms units),
teacher, lr, batch, max_steps, prune_step, ...

Metric: cosine(student step, teacher embedding) on speech-active steps (audio in the teacher's
span >= 10 dB above the window's quietest 50 ms), on
  real_val  real recordings (AMI far-field, LibriSpeech dev-clean, MUSAN speech): SELECTION
  sim_val   multi_test, held-out voices in the simulator test mixer: reported
  word_acc  66-word accuracy of a small head on the student's outputs (multi_train -> multi_test,
            probe_teachers.head_acc), once on the best weights: comparable across teachers
Training data is simulated only. real_test is never read here (finalists only, separately).
Pruning: at prune_step a run continues only if its real_val cosine is in the top third of the
runs in its group that reached that step (needs >= 3 of them).
"""

import argparse
import hashlib
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import frontend_var as FV
import search_space
import store
import students as S

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "data/probe"
RUNS = ROOT / "data/stage_a/runs"
BOOK = ROOT / "research/stage_a"
QUEUE, LEDGER = BOOK / "queue.jsonl", BOOK / "ledger.jsonl"
DEV = "cuda"
EVAL_EVERY = 500
LEAN5 = dict(pcm_bits=12, coef_frac=8, y_w=18, log_frac=2, acc_w=20, acc_shift=4, feat_bits=6)

DEFAULTS = {
    "group": "misc", "size": 50_000, "family": "tcn", "knobs": {},
    "bands": 12, "order": 4, "hop_ms": 16, "stride": 2, "weight_bits": 8,
    "teacher": "whisper_tiny_L4", "train_sets": ["windows_telescope", "multi_train", "windows_computer", "windows_marvin",
                                                  "windows_destiny", "windows_shepherd", "windows_tourists"],
    "lr": 2e-3, "weight_decay": 1e-2, "warmup": 200, "batch": 384, "loss_cos": 0.5,
    "max_steps": 10_000, "prune_step": 3000, "seed": 0,
}

# Teacher embeddings of each audio set. speech_embedding: probe_embed.py (data/probe/emb, the
# telescope windows are stored as "windows"); others: teach_export.py, data/probe/distill/<teacher>/<set>.
def teacher_path(teacher, s):
    if teacher == "speech_embedding":
        return PROBE / "emb" / ("windows" if s == "windows_telescope" else s)
    return PROBE / "distill" / teacher / s


def teacher_ends(teacher, n):
    """Last 16 kHz sample (from the window start) covered by teacher step j."""
    j = np.arange(n)
    if teacher == "speech_embedding":
        return (8 * j + 75) * 160 + 400
    return (j + 10) * 1280 - 1  # 80 ms pooled bins 9..19 (teach_export.py)


# --- config / ids -------------------------------------------------------------------------

def full(cfg):
    c = {**DEFAULTS, **cfg}
    c["knobs"] = dict(sorted(c["knobs"].items()))
    return c


def run_id(cfg):
    c = {k: v for k, v in full(cfg).items() if k not in ("name", "group")}
    return hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:10]


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).exists() else []


def append_jsonl(p, row):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row) + "\n")


# --- data ---------------------------------------------------------------------------------

def fe_variant(c):
    return FV.variant(**LEAN5, bands=c["bands"], order=c["order"], hop=c["hop_ms"] * 16)


def fe_name(c):
    return f"sa_b{c['bands']}_o{c['order']}_h{c['hop_ms']}"


def features(c, s):
    """Cached chip features (N, frames, bands) uint8 codes of audio set s."""
    p = PROBE / "feat" / fe_name(c) / s
    if not store.exists(p):
        v = fe_variant(c)
        A = store.load(PROBE / s)["audio"]
        store.save(p, X=np.concatenate([FV.features(np.asarray(A[i : i + 4096]), v) for i in range(0, len(A), 4096)]))
    return np.asarray(store.load(p)["X"])


def active_mask(audio, ends, span=12400, floor_ms=50, db=10.0):
    """(N, K) bool: teacher step j is speech-active if the RMS over its span (default 0.775 s)
    is >= db above the window's quietest 50 ms."""
    a = audio.astype(np.float32)
    fl = int(16 * floor_ms)
    e = (a[:, : a.shape[1] // fl * fl].reshape(len(a), -1, fl) ** 2).mean(-1)
    floor = 10 * np.log10(e.min(1) + 1e-3)
    out = np.zeros((len(a), len(ends)), bool)
    for j, end in enumerate(ends):
        lo = max(0, end - span)
        out[:, j] = 10 * np.log10((a[:, lo:end] ** 2).mean(1) + 1e-3) >= floor + db
    return out


_cache = {}


def dataset(c):
    """GPU tensors for training and the two validation sets, cached across runs that share
    front end + teacher + train sets."""
    key = (fe_name(c), c["teacher"], tuple(c["train_sets"]))
    if key in _cache:
        return _cache[key]
    _cache.clear()
    t0 = time.time()
    Xs, Ts = [], []
    for s in c["train_sets"]:
        tp = teacher_path(c["teacher"], s)
        if not store.exists(tp):
            print(f"  skip {s}: no {c['teacher']} targets", file=sys.stderr)
            continue
        Xs.append(features(c, s))
        Ts.append(np.asarray(store.load(tp)["X"]).astype(np.float16))
    X, T = np.concatenate(Xs), np.concatenate(Ts)
    v = fe_variant(c)
    sub = np.random.default_rng(0).choice(len(X), min(len(X), 20000), replace=False)
    Xf = FV.to_log2(X[sub], v).reshape(-1, X.shape[-1])
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-3
    Tf = T[sub].reshape(-1, T.shape[-1]).astype(np.float32)
    emu, esd = Tf.mean(0), Tf.std(0) + 1e-3
    D = {"mu": torch.from_numpy(mu).to(DEV), "sd": torch.from_numpy(sd).to(DEV),
         "X": torch.from_numpy(X).to(DEV), "T": torch.from_numpy((T - emu) / esd).to(DEV, torch.float16),
         "n_in": X.shape[-1], "K": T.shape[1], "D": T.shape[-1], "train_windows": len(X)}
    for name, s in (("sim_val", "multi_test"), ("real_val", "real_val")):
        Tv = np.asarray(store.load(teacher_path(c["teacher"], s))["X"]).astype(np.float32)
        audio = np.asarray(store.load(PROBE / s)["audio"])
        D[name] = {"X": torch.from_numpy(features(c, s)).to(DEV), "T": torch.from_numpy((Tv - emu) / esd).to(DEV),
                   "active": torch.from_numpy(active_mask(audio, teacher_ends(c["teacher"], Tv.shape[1]))).to(DEV)}
        if name == "real_val":
            D[name]["src"] = torch.from_numpy(np.asarray(store.load(PROBE / s)["src"])).to(DEV)
    print(f"  data {key[0]} / {key[1]}: {len(X)} train windows, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    _cache[key] = D
    return D


def to_input(D, x):
    """uint8 codes -> standardised log2 features (FV.to_log2 on the GPU)."""
    v = D["v"]
    lo, _ = FV.code_range(v)
    return ((x.float() + lo) / (1 << v["log_frac"]) - D["mu"]) / D["sd"]


# --- run ------------------------------------------------------------------------------------

class Stop(Exception):
    pass


_stop = {"flag": False}


def _on_signal(sig, frame):
    _stop["flag"] = True
    print(f"signal {sig}: will checkpoint and stop", file=sys.stderr, flush=True)


def evaluate(model, D, idx):
    model.eval()
    out = {}
    with torch.no_grad():
        for name in ("real_val", "sim_val"):
            V = D[name]
            cs = []
            for j in range(0, len(V["X"]), 1024):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = model(to_input(D, V["X"][j : j + 1024])).float()[:, idx]
                cs.append(F.cosine_similarity(y, V["T"][j : j + 1024], dim=-1))
            cs = torch.cat(cs)
            act = V["active"]
            out[f"{name}_cos"] = round(float(cs[act].mean()), 4)
            out[f"{name}_cos_all"] = round(float(cs.mean()), 4)
            if "src" in V:
                for k, nm in enumerate(("ami", "libri", "musan")):
                    m = act & (V["src"] == k)[:, None]
                    out[f"{name}_cos_{nm}"] = round(float(cs[m].mean()), 4)
    model.train()
    return out


def word_acc(c, model, D, idx):
    """66-word accuracy of the probe head on the student's outputs (multi_train -> multi_test)."""
    import probe_teachers as PT

    model.eval()
    Y = []
    with torch.no_grad():
        for s in ("multi_train", "multi_test"):
            X = torch.from_numpy(features(c, s)).to(DEV)
            Y.append(torch.cat([model(to_input(D, X[j : j + 1024])).float()[:, idx] for j in range(0, len(X), 1024)])
                     .half().cpu().numpy())
    model.train()
    tags = [np.asarray(store.load(PROBE / s)["tag"]) for s in ("multi_train", "multi_test")]
    return PT.head_acc(Y[0], tags[0], Y[1], tags[1])[0]


def run(cfg):
    c = full(cfg)
    rid = run_id(cfg)
    rdir = RUNS / rid
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "config.json").write_text(json.dumps(c, indent=1))
    D = dataset(c)
    D["v"] = fe_variant(c)
    hop_ms = c["hop_ms"]
    stack = max(1, 16 * c["stride"] // hop_ms)
    torch.manual_seed(c["seed"])
    model, width = S.build(c["family"], c["knobs"], D["n_in"], stack, D["D"], c["size"])
    macs = S.macs_per_step(model, D["n_in"])
    model = model.to(DEV).train()
    n_par = S.n_params(model)
    fsamp = hop_ms * 16
    idx = torch.from_numpy((teacher_ends(c["teacher"], D["K"]) // fsamp) // stack).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / c["warmup"]) * 0.5 * (1 + math.cos(math.pi * s / c["max_steps"])))
    step, curve, best, wall = 0, [], {"real_val_cos": -1.0}, 0.0
    ck = rdir / "ckpt.pt"
    if ck.exists():
        st = torch.load(ck, map_location=DEV, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng"].cpu())
        torch.cuda.set_rng_state(st["cuda_rng"].cpu())
        step, curve, best, wall = st["step"], st["curve"], st["best"], st["wall"]
        print(f"  resume {rid} at step {step}", file=sys.stderr, flush=True)

    def save_ckpt():
        tmp = ck.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "step": step,
                    "curve": curve, "best": best, "wall": wall}, tmp)
        tmp.replace(ck)

    w = c["loss_cos"]
    N = len(D["X"])
    t0 = time.time()
    status = "done"
    while step < c["max_steps"]:
        i = torch.randint(N, (c["batch"],), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(to_input(D, D["X"][i])).float()[:, idx]
        t = D["T"][i].float()
        loss = 2 * ((1 - w) * F.mse_loss(y, t) + w * (1 - F.cosine_similarity(y, t, dim=-1)).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % EVAL_EVERY == 0 or step == c["max_steps"]:
            wall += time.time() - t0
            t0 = time.time()
            m = evaluate(model, D, idx)
            curve.append({"step": step, "loss": round(loss.item(), 4), "wall_s": round(wall), **m})
            append_jsonl(rdir / "curve.jsonl", curve[-1])
            print(f"  {c.get('name', rid)} step {step} real {m['real_val_cos']:.3f} sim {m['sim_val_cos']:.3f} ({wall:.0f}s)",
                  file=sys.stderr, flush=True)
            if m["real_val_cos"] > best["real_val_cos"]:
                best = {"step": step, **m}
                torch.save(model.state_dict(), rdir / "best.pt")
            if step == c["prune_step"] and prune(c, m["real_val_cos"]):
                status = "pruned"
                save_ckpt()
                break
            save_ckpt()
            if _stop["flag"]:
                raise Stop
    if status == "done":
        model.load_state_dict(torch.load(rdir / "best.pt", map_location=DEV))
        best["word_acc"] = word_acc(c, model, D, idx)
    at_prune = next((r["real_val_cos"] for r in curve if r["step"] == c["prune_step"]), None)
    v = fe_variant(c)
    row = {"id": rid, "name": c.get("name", rid), "group": c["group"], "status": status, "date": time.strftime("%Y-%m-%d %H:%M"),
           "config": c, "params": n_par, "width": width, "rf_s": round(model.body.rf_steps * 0.016 * stack * hop_ms / 16, 2)
           if math.isfinite(model.body.rf_steps) else None,
           "frontend_ff": FV.flipflops(v), "cycles_per_sample": FV.cycles_per_sample(v),
           "qspi_bytes_per_step": n_par * c["weight_bits"] // 8, "macs_per_step": macs,
           "fits_budget": n_par <= _budget(c) and macs <= search_space.MACS_PER_FRAME * c["stride"],
           "train_windows": D["train_windows"], "steps": step, "wall_s": round(wall),
           "cos_at_prune": at_prune, "best": best, "final": curve[-1] if curve else None}
    append_jsonl(LEDGER, row)
    print(f"  -> {status}: {row['name']} real_val_cos {best['real_val_cos']} word_acc {best.get('word_acc')} params {n_par}", file=sys.stderr, flush=True)
    return row


def _budget(c):
    return search_space.budget(c["weight_bits"], c["stride"])


def prune(c, cos):
    peers = [r["cos_at_prune"] for r in read_jsonl(LEDGER) if r["group"] == c["group"] and r.get("cos_at_prune") is not None]
    if len(peers) < 3:
        return False
    return cos < np.quantile(peers, 2 / 3)


# --- queue ----------------------------------------------------------------------------------

def add(path):
    have = {run_id(q) for q in read_jsonl(QUEUE)}
    n = 0
    for cfg in read_jsonl(path):
        if run_id(cfg) not in have:
            append_jsonl(QUEUE, cfg)
            have.add(run_id(cfg))
            n += 1
    print(f"queued {n} new configs ({len(have)} total)")


def pending():
    done = {r["id"] for r in read_jsonl(LEDGER)}
    q = [cfg for cfg in read_jsonl(QUEUE) if run_id(cfg) not in done]
    started = [cfg for cfg in q if (RUNS / run_id(cfg) / "ckpt.pt").exists()]
    return started + [cfg for cfg in q if cfg not in started]


def work(max_runs):
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    n = 0
    while n < max_runs:
        q = pending()
        if not q:
            print("queue empty", file=sys.stderr)
            return
        cfg = q[0]
        print(f"run {cfg.get('name')} [{run_id(cfg)}] ({len(q)} pending)", file=sys.stderr, flush=True)
        try:
            run(cfg)
        except Stop:
            print("stopped; resume with `work`", file=sys.stderr)
            return
        except NotImplementedError as e:
            append_jsonl(LEDGER, {"id": run_id(cfg), "name": cfg.get("name"), "group": full(cfg)["group"],
                                  "status": f"skipped: not implemented ({e})", "config": full(cfg)})
        n += 1


def rescore():
    """Add word_acc to finished ledger runs that lack it (rebuilds each student, loads best.pt)."""
    rows = read_jsonl(LEDGER)
    for r in rows:
        b = r.get("best") or {}
        if r.get("status") != "done" or "word_acc" in b or not (RUNS / r["id"] / "best.pt").exists():
            continue
        c = r["config"]
        D = dataset(c)
        D["v"] = fe_variant(c)
        stack = max(1, 16 * c["stride"] // c["hop_ms"])
        model, _ = S.build(c["family"], c["knobs"], D["n_in"], stack, D["D"], c["size"])
        model = model.to(DEV)
        model.load_state_dict(torch.load(RUNS / r["id"] / "best.pt", map_location=DEV))
        idx = torch.from_numpy((teacher_ends(c["teacher"], D["K"]) // (c["hop_ms"] * 16)) // stack).to(DEV)
        b["word_acc"] = word_acc(c, model, D, idx)
        print(f"{r['name']}: word_acc {b['word_acc']}", file=sys.stderr, flush=True)
    LEDGER.write_text("".join(json.dumps(r) + "\n" for r in rows))


def status():
    L = read_jsonl(LEDGER)
    print(f"ledger: {len(L)} runs; pending: {len(pending())}")
    for r in L[-30:]:
        b = r.get("best") or {}
        print(f"{r['name'][:40]:40s} {r['status'][:8]:8s} params {r.get('params')}  real {b.get('real_val_cos')}  sim {b.get('sim_val_cos')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["add", "work", "status", "rescore"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--max-runs", type=int, default=10**9)
    a = ap.parse_args()
    {"add": lambda: add(a.path), "work": lambda: work(a.max_runs), "status": status, "rescore": rescore}[a.what]()
