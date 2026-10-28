"""Stage A as one SMAC3 study: the optimizer picks front end, engine, family + knobs, teacher and
distillation settings; the network size is derived so every trial fits the 16-tile chip.

  python model/stage_a_smac.py work      # run trials until stopped; SIGTERM / Ctrl-C checkpoints and exits
  python model/stage_a_smac.py status    # best configs so far
  python model/stage_a_smac.py space     # print the search space and the fit of a few configs

Why SMAC3 (MultiFidelityFacade): random-forest surrogate over a conditional mixed space (each
family has its own knobs), Hyperband over training steps (most trials stop at 1000 steps, the
best are continued to 3000 and 9000), and it treats the objective as noisy. Training seeds are
not fixed: each run draws a fresh one. Every NOISE_EVERY trials the current incumbent is re-run
at the full budget with a new seed; the spread of those repeats is our measured run-to-run noise
(research/stage_a/smac_trials.jsonl, kind "repeat").

Objective: maximise word_acc (66-word accuracy of the probe head trained on the student's
outputs, held-out voices in the simulator test mixer; stage_a.word_acc). It is comparable across
teachers, which cosine is not. Real-audio cosine to the trial's teacher (real_val) and the
simulated-val cosine are logged for every trial as the sim-to-real check.

Fit by construction (derive, don't sample; search_space.py section 1):
  front-end tiles = 5 + (flip-flops - 1,616) / 320 (the 5-tile estimate is 16 bands, 4th order)
  on-chip bytes   = (16 - front-end tiles - 2 engine) * 40
  QSPI bytes/step = params * weight_bits / 8 + state traffic (conv history, KV cache, recurrent
                    state: all in PSRAM) + activation spill (2x all activations when the largest
                    single activation vector does not fit on chip)  <= 65,536 * stride
  MACs/step       <= 262,144 * stride
The param target starts at min(budget(weight_bits, stride), SIZE_CAP) and shrinks by 15% until
all of that holds. SIZE_CAP (262k) keeps a trial under ~1 h at 9000 steps on the shared GPU.
Receptive field must reach 0.8 s (the teacher's window); configs that don't are reported to SMAC
as crashed (cost 0), so it learns to avoid them at no training cost.

Resumable: SMAC's state is in data/stage_a/smac/ (saved on every tell). The trial in flight is in
data/stage_a/smac/pending.json; each (config, repeat) trains in data/stage_a/smac/runs/<id>/ and
a promotion to a larger budget continues that checkpoint (the LR schedule spans MAX_STEPS, so a
1000-step result is mid-schedule, like an early-stopping check).
"""

import argparse
import hashlib
import json
import math
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from ConfigSpace import (Categorical, Configuration, ConfigurationSpace, EqualsCondition, Float, ForbiddenAndConjunction,
                         ForbiddenEqualsClause, Integer, OrdinalHyperparameter)

import frontend_var as FV
import search_space as SS
import stage_a as SA
import students as S

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data/stage_a/smac"
RUNS = WORK / "runs"
PENDING = WORK / "pending.json"
JOURNAL = ROOT / "research/stage_a/smac_trials.jsonl"
DEV = "cuda"

MIN_STEPS, MAX_STEPS, ETA = 1000, 9000, 3
SIZE_CAP = 262_144
NOISE_EVERY = 10
MIN_RF_S = 0.8
FE_FF_REF, FE_TILES_REF = 1616, 5.0  # lean5, 16 bands, 4th order = the 5-tile front-end estimate
BUS_PER_FRAME, MACS_PER_FRAME = 65_536, 262_144

# Family knobs not implemented in students.py are left out of the space.
DROP = {"rnn": {"proj"}, "svdf": "all"}
RESTRICT = {"ssm.kind": ["lru", "mingru"], "transformer.pos": ["rel_bias", "none"]}
FAMILIES = ["tcn", "bcresnet", "rnn", "ssm", "transformer", "crnn", "dnn"]
TEACHERS = ["speech_embedding", "whisper_tiny_L4", "whisper_tiny_L4_causal"]


# --- search space ---------------------------------------------------------------------------

def _hp(name, spec):
    kind = spec[0]
    if kind == "cat":
        vals = RESTRICT.get(name, list(spec[1]))
        if all(isinstance(v, bool) for v in vals):
            return Categorical(name, [str(v) for v in vals])
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals) and len(vals) > 2:
            return OrdinalHyperparameter(name, sorted(vals))
        return Categorical(name, vals)
    if kind == "int":
        return Integer(name, (spec[1], spec[2]))
    return Float(name, (spec[1], spec[2]), log=kind == "log")


def space():
    cs = ConfigurationSpace()
    # Front end: A1 settled 4th order (+3 pts word_acc at every band count, research/07_stage_a).
    cs.add([_hp("bands", SS.FRONTEND["bands"]), Categorical("hop_ms", [16, 32]),
            Categorical("weight_bits", [8, 4]), OrdinalHyperparameter("stride", [1, 2, 4]),
            Categorical("family", FAMILIES), Categorical("teacher", TEACHERS),
            Float("loss_cos", (0.0, 1.0)), Float("lr", (3e-4, 1e-2), log=True),
            Float("weight_decay", (1e-4, 1e-1), log=True), OrdinalHyperparameter("warmup", [100, 300, 1000]),
            OrdinalHyperparameter("batch", [256, 384, 512]), OrdinalHyperparameter("grad_clip", [0.5, 1.0, 5.0]),
            Float("aug_gain_db", (0.0, 12.0)), Float("aug_tilt", (0.0, 1.0)), Float("aug_band_mask_p", (0.0, 0.4)),
            Float("qat_start", (0.3, 0.8))])
    for fam in FAMILIES:
        for k, spec in SS.FAMILIES[fam].items():
            if k in DROP.get(fam, ()):
                continue
            hp = _hp(f"{fam}.{k}", spec)
            cs.add(hp)
            cs.add(EqualsCondition(hp, cs["family"], fam))
    # hop 32 ms cannot produce 16 ms output steps
    cs.add(ForbiddenAndConjunction(ForbiddenEqualsClause(cs["stride"], 1), ForbiddenEqualsClause(cs["hop_ms"], 32)))
    return cs


def BASELINE(cs):
    """The A1 reference (12 bands, 50k-class TCN, speech_embedding, stage_a DEFAULTS) as the first trial."""
    return Configuration(cs, values={"bands": 12, "hop_ms": 16, "weight_bits": 8, "stride": 2, "family": "tcn",
                                     "teacher": "speech_embedding", "loss_cos": 0.5, "lr": 2e-3, "weight_decay": 1e-2,
                                     "warmup": 300, "batch": 384, "grad_clip": 1.0, "aug_gain_db": 0.0, "aug_tilt": 0.0,
                                     "aug_band_mask_p": 0.0, "qat_start": 0.6, "tcn.depth": 8, "tcn.kernel": 3,
                                     "tcn.dilation": "exp_cycle4", "tcn.separable": "True", "tcn.gated": "False",
                                     "tcn.norm": "bn", "tcn.expand": 1})


def to_cfg(conf):
    """ConfigSpace configuration -> the stage_a config dict (+ trial knobs)."""
    d = dict(conf)
    fam = d["family"]
    knobs = {}
    for k, v in d.items():
        if k.startswith(fam + "."):
            v = {"True": True, "False": False}.get(v, v) if isinstance(v, str) else v
            knobs[k.split(".", 1)[1]] = v.item() if hasattr(v, "item") else v
    c = {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items() if "." not in k}
    c.update({"knobs": knobs, "order": 4, "train_sets": SA.DEFAULTS["train_sets"]})
    return c


# --- fit: derive the size ---------------------------------------------------------------------

def hw(model, n_in, stack):
    """Per output step: state traffic (bytes), largest activation vector (bytes), all activations
    (bytes), with 8-bit activations. Traced with hooks on a T-step forward pass."""
    T = 37
    out = {"state": 0, "act_max": 0, "act_sum": 0}

    def hook(mod, inp, o):
        x = inp[0] if isinstance(inp, tuple) and inp else None
        o = o[0] if isinstance(o, tuple) else o
        if isinstance(o, torch.Tensor) and o.dim() >= 3:
            L = o.shape[-1] if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d)) else o.shape[1]
            if L in (T, T * stack):
                out["act_max"] = max(out["act_max"], o[0].numel() // L)
                out["act_sum"] += o[0].numel() // T
        if isinstance(mod, (nn.Conv1d, nn.Conv2d)) and x is not None:
            kt = mod.kernel_size[-1]
            if kt > 1:
                units = o.shape[-1] / T  # time units per output step (frames for BC-ResNet)
                out["state"] += int(x[0].numel() / x.shape[-1] * kt * units)
        elif type(mod).__name__ == "_Attn" and x is not None:
            out["state"] += (2 * mod.w + 2) * x.shape[-1]
        elif isinstance(mod, (nn.GRU, nn.LSTM)):
            out["state"] += 2 * mod.hidden_size * mod.num_layers * (2 if isinstance(mod, nn.LSTM) else 1)
        elif type(mod).__name__ in ("_MinGRU", "_LRU") and isinstance(o, torch.Tensor):
            out["state"] += 2 * o[0].numel() // T
        elif type(mod).__name__ == "DNN" and x is not None:
            out["state"] += mod.context * x.shape[-1]

    hs = [m.register_forward_hook(hook) for m in model.modules()]
    with torch.no_grad():
        model.eval()(torch.zeros(1, T * stack, n_in))
    for h in hs:
        h.remove()
    return out


def fit(c, n_in):
    """Largest student of this config that fits -> (model, width, report). None if nothing fits."""
    v = SA.fe_variant(c)
    ff = FV.flipflops(v)
    fe_tiles = FE_TILES_REF + (ff - FE_FF_REF) / SS.TILE_BITS
    on_chip = (SS.TILES - fe_tiles - SS.ENGINE_TILES) * SS.TILE_BITS / 8
    stack = max(1, 16 * c["stride"] // c["hop_ms"])
    target = min(SS.budget(c["weight_bits"], c["stride"]), SIZE_CAP)
    for _ in range(20):
        model, width = S.build(c["family"], c["knobs"], n_in, stack, 96, target)
        p = S.n_params(model)
        h = hw(model, n_in, stack)
        spill = 2 * h["act_sum"] if h["act_max"] > on_chip / 2 else 0  # in + out buffers on chip
        bus = p * c["weight_bits"] / 8 + h["state"] + spill
        macs = S.macs_per_step(model, n_in)
        rep = {"params": p, "width": width, "frontend_ff": ff, "frontend_tiles": round(fe_tiles, 2),
               "on_chip_bytes": int(on_chip), "act_max_bytes": h["act_max"], "state_bytes_per_step": h["state"],
               "spill_bytes_per_step": spill, "bus_bytes_per_step": int(bus), "bus_limit": BUS_PER_FRAME * c["stride"],
               "macs_per_step": macs, "mac_limit": MACS_PER_FRAME * c["stride"], "param_target": target,
               "rf_s": round(model.body.rf_steps * 0.016 * c["stride"], 2) if math.isfinite(model.body.rf_steps) else None}
        if bus <= BUS_PER_FRAME * c["stride"] and macs <= MACS_PER_FRAME * c["stride"] and fe_tiles + SS.ENGINE_TILES < SS.TILES:
            return model, width, rep
        target = int(target * 0.85)
        if target < 4000:
            break
    return None, None, rep


# --- training -------------------------------------------------------------------------------

class _FakeQuant(nn.Module):
    """Symmetric per-output-channel weight fake-quant, straight-through gradient; off until enabled."""

    def __init__(self, bits):
        super().__init__()
        self.q = 2 ** (bits - 1) - 1
        self.register_buffer("on", torch.zeros(()))

    def forward(self, w):
        if not self.on:
            return w
        s = w.detach().abs().flatten(1).amax(1).clamp(min=1e-8).view(-1, *[1] * (w.dim() - 1)) / self.q
        return w + (torch.round(w / s).clamp(-self.q, self.q) * s - w).detach()


def add_qat(model, bits):
    fqs = []
    for mod in list(model.modules()):
        for name, p in list(mod.named_parameters(recurse=False)):
            if p.dim() >= 2:
                fq = _FakeQuant(bits)
                P.register_parametrization(mod, name, fq)
                fqs.append(fq)
    return fqs


def train_input(D, x, c, g):
    """uint8 codes -> standardised log2 features, with the trial's augmentations (student only)."""
    v = D["v"]
    lo, _ = FV.code_range(v)
    l2 = (x.float() + lo) / (1 << v["log_frac"])
    B, _, nb = l2.shape
    if c["aug_gain_db"] > 0:  # level shift: log2 units = dB / 6.02
        l2 = l2 + (torch.rand(B, 1, 1, device=DEV, generator=g) * 2 - 1) * c["aug_gain_db"] / 6.02
    if c["aug_tilt"] > 0:  # spectral tilt across the bands, +-aug_tilt log2 units end to end
        ramp = torch.linspace(-0.5, 0.5, nb, device=DEV)
        l2 = l2 + (torch.rand(B, 1, 1, device=DEV, generator=g) * 2 - 1) * c["aug_tilt"] * ramp
    z = (l2 - D["mu"]) / D["sd"]
    if c["aug_band_mask_p"] > 0:  # drop whole bands (to the mean)
        z = z * (torch.rand(B, 1, nb, device=DEV, generator=g) >= c["aug_band_mask_p"])
    return z


_stop = {"flag": False}


def _on_signal(sig, frame):
    _stop["flag"] = True
    print(f"signal {sig}: will checkpoint and stop", file=sys.stderr, flush=True)


class Stop(Exception):
    pass


def run_dir(conf, rep):
    key = json.dumps(dict(sorted((k, str(v)) for k, v in dict(conf).items())), sort_keys=True)
    return RUNS / (hashlib.sha1(key.encode()).hexdigest()[:10] + f"_r{rep}")


def train(conf, budget, rep):
    """Train (or continue) config `conf`, repeat `rep`, to `budget` steps -> metrics row."""
    c = to_cfg(conf)
    rdir = run_dir(conf, rep)
    rdir.mkdir(parents=True, exist_ok=True)
    D = SA.dataset(c)
    D["v"] = SA.fe_variant(c)
    stack = max(1, 16 * c["stride"] // c["hop_ms"])
    st_path = rdir / "ckpt.pt"
    st = torch.load(st_path, map_location=DEV, weights_only=False) if st_path.exists() else None
    seed = st["seed"] if st else random.randrange(2**31)
    torch.manual_seed(seed)
    model, width, rep_hw = fit(c, D["n_in"])
    row = {"config": c, "budget": budget, "repeat": rep, "seed": seed, **rep_hw}
    if model is None:
        return {**row, "status": "does_not_fit"}
    if rep_hw["rf_s"] is not None and rep_hw["rf_s"] < MIN_RF_S:
        return {**row, "status": "rf_too_short"}
    model = model.to(DEV).train()
    fqs = add_qat(model, c["weight_bits"])
    idx = torch.from_numpy((SA.teacher_ends(c["teacher"], D["K"]) // (c["hop_ms"] * 16)) // stack).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / c["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(s, MAX_STEPS) / MAX_STEPS)))
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    step, wall, curve = 0, 0.0, []
    if st:
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        g.set_state(st["gen"].cpu())
        step, wall, curve = st["step"], st["wall"], st["curve"]
        print(f"  resume {rdir.name} at step {step}", file=sys.stderr, flush=True)

    def save():
        tmp = st_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "gen": g.get_state(), "seed": seed, "step": step, "wall": wall, "curve": curve}, tmp)
        tmp.replace(st_path)

    qat_at = int(c["qat_start"] * MAX_STEPS)
    w = c["loss_cos"]
    N = len(D["X"])
    t0 = time.time()
    while step < budget:
        for fq in fqs:
            fq.on.fill_(float(step >= qat_at))
        i = torch.randint(N, (c["batch"],), device=DEV, generator=g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(train_input(D, D["X"][i], c, g)).float()[:, idx]
        t = D["T"][i].float()
        loss = 2 * ((1 - w) * F.mse_loss(y, t) + w * (1 - F.cosine_similarity(y, t, dim=-1)).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
        opt.step()
        sched.step()
        step += 1
        if step % 500 == 0 or step == budget:
            wall += time.time() - t0
            t0 = time.time()
            curve.append({"step": step, "loss": round(loss.item(), 4), "wall_s": round(wall)})
            save()
            if _stop["flag"]:
                raise Stop
    for fq in fqs:  # evaluate as the chip runs it once QAT has started
        fq.on.fill_(float(step >= qat_at))
    m = SA.evaluate(model, D, idx)
    m["word_acc"] = SA.word_acc(c, model, D, idx)
    return {**row, "status": "ok", "steps": step, "wall_s": round(wall), "quantized": step >= qat_at, **m}


# --- SMAC loop ------------------------------------------------------------------------------

def facade():
    from smac import MultiFidelityFacade as MF
    from smac import Scenario

    cs = space()
    sc = Scenario(cs, name="stage_a", output_directory=WORK / "smac3", deterministic=False, n_trials=100_000,
                  min_budget=MIN_STEPS, max_budget=MAX_STEPS, seed=0)
    smac = MF(sc, target_function=lambda config, seed, budget: 0.0, overwrite=False, logging_level=30,
              intensifier=MF.get_intensifier(sc, eta=ETA),
              initial_design=MF.get_initial_design(sc, n_configs=12, additional_configs=[BASELINE(cs)]))
    return smac, cs


def read_journal():
    return [json.loads(l) for l in JOURNAL.read_text().splitlines() if l.strip()] if JOURNAL.exists() else []


def work():
    from smac.runhistory import StatusType, TrialInfo, TrialValue

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    WORK.mkdir(parents=True, exist_ok=True)
    smac, cs = facade()
    while True:
        if PENDING.exists():
            p = json.loads(PENDING.read_text())
            info = TrialInfo(config=Configuration(cs, values=p["config"]), seed=p["seed"], budget=p["budget"])
            kind, rep = p["kind"], p["repeat"]
        else:
            kinds = [r.get("kind") for r in read_journal()]
            since = len(kinds) - 1 - max((i for i, k in enumerate(kinds) if k == "repeat"), default=-1)
            inc = smac.intensifier.get_incumbent()
            if since >= NOISE_EVERY and inc is not None:
                rep = 1 + kinds.count("repeat")
                info = TrialInfo(config=inc, seed=random.randrange(2**31), budget=MAX_STEPS)
                kind = "repeat"
            else:
                info, kind, rep = smac.ask(), "trial", 0
            PENDING.write_text(json.dumps({"config": {k: (v.item() if hasattr(v, "item") else v) for k, v in dict(info.config).items()},
                                           "seed": info.seed, "budget": info.budget, "kind": kind, "repeat": rep}))
        c = to_cfg(info.config)
        print(f"{kind} {c['family']} b{c['bands']} h{c['hop_ms']} s{c['stride']} w{c['weight_bits']} {c['teacher']} "
              f"budget {int(info.budget)}", file=sys.stderr, flush=True)
        try:
            row = train(info.config, int(info.budget), rep)
        except Stop:
            print("stopped; resume with `work`", file=sys.stderr)
            return
        except (NotImplementedError, RuntimeError) as e:
            row = {"config": c, "budget": info.budget, "status": f"error: {type(e).__name__}: {str(e)[:200]}"}
            torch.cuda.empty_cache()
        ok = row["status"] == "ok"
        smac.tell(info, TrialValue(cost=-row["word_acc"] if ok else 0.0,
                                   status=StatusType.SUCCESS if ok else StatusType.CRASHED))
        row = {"n": len(read_journal()), "kind": kind, "date": time.strftime("%Y-%m-%d %H:%M"), **row}
        SA.append_jsonl(JOURNAL, row)
        PENDING.unlink()
        print(f"  -> {row['status']} word_acc {row.get('word_acc')} real {row.get('real_val_cos')} params {row.get('params')}",
              file=sys.stderr, flush=True)


def status():
    J = read_journal()
    ok = [r for r in J if r.get("status") == "ok"]
    print(f"{len(J)} trials, {len(ok)} ok; by budget: " +
          ", ".join(f"{b}: {sum(r['budget'] == b for r in ok)}" for b in sorted({r['budget'] for r in ok})))
    reps = [r["word_acc"] for r in ok if r["kind"] == "repeat"]
    if len(reps) > 1:
        print(f"repeats of the incumbent at {MAX_STEPS}: word_acc sd {np.std(reps, ddof=1):.4f} (n={len(reps)})")
    for r in sorted(ok, key=lambda r: -r["word_acc"])[:15]:
        c = r["config"]
        print(f"{r['word_acc']:.4f} real {r['real_val_cos']:.3f} {c['family']:11s} b{c['bands']:<2d} h{c['hop_ms']} s{c['stride']} "
              f"w{c['weight_bits']} {c['teacher'][:22]:22s} params {r['params']:6d} steps {r['steps']}")


def show_space():
    cs = space()
    print(cs)
    for conf in [BASELINE(cs)] + list(cs.sample_configuration(8)):
        c = to_cfg(conf)
        try:
            _, _, rep = fit(c, c["bands"])
        except NotImplementedError as e:
            rep = str(e)
        print(c["family"], c["bands"], c["hop_ms"], c["stride"], c["weight_bits"], json.dumps(rep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["work", "status", "space"])
    {"work": work, "status": status, "space": show_space}[ap.parse_args().what]()
