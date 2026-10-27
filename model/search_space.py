"""The search space for the architecture + training study, in two stages, for review.

Read this file top to bottom: every searched knob is listed once, with its range and a
one-line reason. Nothing here runs training; `sample_student(trial)` / `sample_detector(trial)`
turn an Optuna trial (or any object with the same suggest_* methods) into a config dict, and
`budget()` is the hard 16-tile constraint every config must meet.

Why two stages (research/06_probes, P5/P6): a student network on chip features, trained to
reproduce Google's speech embedding (the teacher) on simulated audio, plus a small detector
head, misses 31-33% of keywords at 5 FA/h against 61-69% for the same features trained from
scratch. So the network is trained in two steps, and each gets its own search:

  Stage A  STUDENT   chip features -> 96-d teacher embedding. Word-independent. Expensive
                     (10-45 min per trial). Scored by how well it copies the teacher.
  Stage B  DETECTOR  frozen student -> "was that the wake word?". Per word. Cheap (~20 s per
                     head). Scored by FRR at FA/h. Not needed before tapeout.
  Stage C  joint fine-tune of the finalists (student unfrozen, a little distillation loss
           kept), 4-bit QAT if chosen. A handful of runs, no optimizer.

What tapeout freezes (Tiny Tapeout shuttle, ~2026-11-30): the weights live in QSPI flash and
can be retrained after tapeout, so Stage A only has to settle the HARDWARE section below
(front end, weight precision, output stride, which layer types the engine runs, where state
lives) and show the best students fit the budget. Fine-grained tuning can wait.

Stage A metric (decided 2026-10-27): cosine between student output and teacher embedding,
on speech-bearing windows only (teacher-energy gate, so silence/music copying doesn't count).
  * Training audio: simulated only (TTS conversation + ambience + rooms, train mixer).
  * Selection (val): REAL audio. Distillation needs no labels, so any real recording is a
    valid target: teacher(real audio) vs student(chip features of the same audio). See
    STAGE_A_EVAL. This measures the sim-to-real gap directly; the simulated val (held-out
    voices) is reported next to it.
  * Final test: a disjoint set of real recordings (other corpus halves), read once per finalist.
  * Guard against proxy drift: every ~10 trials, train a Stage B head on the current best
    student and check FRR still follows the cosine (it did up to ~0.76: research/06_probes).

Stage A plan (each step fixes what it settles before the next):
  A0 control: same student trained from scratch as a 66-word classifier (no teacher) + head,
     on the same audio. If it matches distillation, the teacher isn't what helps.
  A1 front end: one fixed TCN student, FRONTEND grid (bands x order), 2 sizes.
     Compare cosine against flip-flops (frontend_var.flipflops).
  A2 families: each STAGE_A_FAMILIES family at 2 sizes (SIZES[1], SIZES[3]), ~6 trials each,
     on the A1 front end. Answers which layer types the engine must run.
  A3 size ladder for the top 2 families: all SIZES, both strides. The size/area curve.
  A4 precision: 8-bit vs 4-bit QAT on the A3 winner at equal QSPI bytes.
  A5 distillation knobs (DISTILL) on the winner: optimizer, loss, data, teacher_view.
Trials stop early: rankings are visible by step ~3000 (06_probes distill logs), so a trial
runs 3000 steps unless it is in the top third at that point, then continues to 10000.

Pretrained nets: research/SOA.md section 10. None take our 16 ms IIR features, so they enter
as teachers (`DISTILL.teacher`) and architecture templates, not as copied layers.
"""

import math

# ======================================================================================
# 1. Hardware budget: what "fits in 16 tiles" means
# ======================================================================================
# From archs.cost() (research/02_hardware_and_architectures):
#   tiles = 5 front end + 2 engine + on-chip working bytes * 8 / 320
#   weights stream from QSPI flash every output step; long state (conv history, KV cache,
#   SVDF memories, and optionally the recurrent state) lives in PSRAM on the same QSPI bus.
TILES = 16
FRONTEND_TILES, ENGINE_TILES, TILE_BITS = 5.0, 2.0, 320
ON_CHIP_BYTES = (TILES - FRONTEND_TILES - ENGINE_TILES) * TILE_BITS / 8  # 360 B working set
QSPI_BYTES_PER_FRAME = 4.096e6 * 0.016  # 65.5 kB per 16 ms frame, shared by weights + PSRAM
MACS_PER_FRAME = 16.384e6 * 0.016  # 262k: one 8x8 MAC per clock


def budget(weight_bits, stride):
    """Max parameters for one 16 ms frame budget, at a weight precision and output stride.
    The binding limit is usually QSPI bandwidth:
        8-bit, stride 1:  65k params    4-bit, stride 1: 131k
        8-bit, stride 2: 131k           4-bit, stride 2: 262k
        8-bit, stride 4: 262k           4-bit, stride 4: 524k
    (the one-MAC-per-clock limit is 262k x stride, so it never binds first here)
    PSRAM state traffic is subtracted per model by the builder (it knows its state size).
    Student + detector head together must fit; the head is small (Stage B keeps it <10%)."""
    qspi = QSPI_BYTES_PER_FRAME * stride * 8 / weight_bits
    macs = MACS_PER_FRAME * stride
    return int(min(qspi, macs))


# ("cat", [...])           categorical
# ("int", lo, hi[, step])  integer
# ("float", lo, hi)        uniform
# ("log", lo, hi)          log-uniform
# Family-specific knobs are only sampled for that family.

# ======================================================================================
# 2. HARDWARE: frozen at tapeout (Stage A must settle these)
# ======================================================================================
FRONTEND = {
    # The chip's filterbank (model/frontend_var.py, research/06_probes F2). Each choice costs area,
    # so it is scored against flip-flops, not just accuracy. lean5 bit widths are fixed (free cuts:
    # pcm 12, Q2.8 coefs, state 18, acc 20, 6-bit log codes at 1/4 octave; no loss on the probe).
    "bands": ("cat", [8, 10, 12, 14, 16, 20, 24]),  # mel-spaced bands; ~55 FF/band (lean5, 2nd
    # order), ~90 at 4th; 66-word probe: 8 -> -4.4 pts, 12 (4th) -1.0 vs 16 (4th), 24 -> +1.1
    "order": ("cat", [2, 4]),  # 4 = two cascaded resonators per band: +2.8 pts, +~35 FF/band
    "hop_ms": ("cat", [16, 32]),  # filterbank frame; 32 ms: -0.6 pts, halves the input rate
}
FRONTEND_DEFAULT = {"bands": 12, "order": 4, "hop_ms": 16}  # lean5o4b12: 54.2% at 1,248 FF

ENGINE = {
    "weight_bits": ("cat", [8, 4]),  # 4-bit = QAT, doubles the params the bus allows (A4)
    "stride": ("cat", [1, 2, 4]),  # network output every 16/32/64 ms (frames stacked at the input)
    "state_in_psram": ("cat", [False, True]),  # long state (conv history, recurrent state, KV
    # cache) in PSRAM: frees on-chip bytes, costs bus bandwidth (builder subtracts it from budget)
}
ENGINE_DEFAULT = {"weight_bits": 8, "stride": 2, "state_in_psram": True}

# ======================================================================================
# 3. Stage A: STUDENT (chip features -> teacher embedding)
# ======================================================================================
# Size is held at fixed constants, not searched: the optimizer would otherwise always pick
# the largest model, and smaller is what we want on chip. Small models are not much faster
# on this GPU (45k -> 330k params cost 1.7x wall time), so the ladder is about the area/
# accuracy curve, not speed. Params include the final projection to the teacher's 96 dims.
SIZES = (12_000, 25_000, 50_000, 100_000, 200_000)  # ~2x steps; 8-bit stride 2 allows 131k

STUDENT_COMMON = {
    "in_norm": ("cat", ["global", "running_mean", "pcen_lite"]),  # running per-band mean in the log
    # domain (cheap on chip): level/channel robustness, the classic far-field fix (PCEN)
    "in_norm_tau_s": ("log", 0.3, 5.0),  # time constant for running_mean / pcen_lite
    "in_deltas": ("cat", [False, True]),  # append frame-to-frame differences (bands more inputs)
    "out_proj": ("cat", ["linear", "bottleneck32"]),  # last layer to 96 dims, direct or via 32
}
# Constraint (builder): receptive field >= 0.8 s, the teacher's window (76 x 10 ms + 25 ms).

FAMILIES = {
    # Apple 2017 first stage [SOA A1]: a DNN over a fixed window of past frames.
    "dnn": {
        "context": ("int", 8, 48),  # frames of history (0.13-0.77 s; Apple: 20 x 10 ms)
        "depth": ("int", 2, 6),
        "act": ("cat", ["sigmoid", "relu", "gelu"]),
        "bottleneck": ("cat", [0, 0.5]),  # optional narrow first layer (fraction of width)
    },
    # GRU / LSTM: tiny state, the classic MCU choice.
    "rnn": {
        "cell": ("cat", ["gru", "lstm"]),
        "layers": ("int", 1, 3),
        "residual": ("cat", [False, True]),
        "proj": ("cat", [0, 0.5]),  # LSTMP-style output projection (Google's LSTM KWS)
    },
    # Linear recurrences trained in parallel, run as a recurrence: S4D / LRU / Mamba-lite.
    # minGRU (one of these) failed earlier only with per-frame norm; retried with in_norm options.
    "ssm": {
        "kind": ("cat", ["lru", "s4d", "mingru", "mamba_lite"]),
        "layers": ("int", 1, 4),
        "state": ("cat", [8, 16, 32, 64]),  # per-channel state size (lru/s4d/mamba)
        "expand": ("cat", [1, 2]),
        "conv_k": ("cat", [0, 3, 4]),  # short causal conv before the scan (Mamba-style)
    },
    # Dilated causal conv stack: TC-ResNet / microWakeWord MixedNet / WaveNet family.
    # The 06_probes student is this family: separable, kernel 3, dilations 1,2,4,8 x2, BN.
    "tcn": {
        "depth": ("int", 3, 10),
        "kernel": ("cat", [3, 5, 7, "mix357"]),  # mix = MixConv: kernels split across channels
        "dilation": ("cat", ["exp", "exp_cycle4", "linear", "none"]),  # 1,2,4,8,.. / restart every 4 / 1,2,3,.. / all 1
        "separable": ("cat", [True, False]),  # depthwise + pointwise vs full conv
        "gated": ("cat", [False, True]),  # WaveNet tanh*sigmoid gate
        "norm": ("cat", ["bn", "none"]),  # BN folds into weights at inference
        "expand": ("cat", [1, 2]),  # inverted bottleneck (MobileNetV2-style) inside each block
    },
    # BC-ResNet (Qualcomm 2021): 2D conv over (freq x time) with a broadcast temporal branch;
    # best accuracy per parameter on Speech Commands at 9k-320k params. Causal in time.
    "bcresnet": {
        "stages": ("int", 2, 4),
        "blocks_per_stage": ("int", 1, 3),
        "freq_stride": ("cat", [1, 2]),
        "subspectral_groups": ("cat", [1, 2, 4]),  # sub-spectral norm groups
    },
    # Google E2E streaming KWS [SOA G2].
    "svdf": {
        "layers": ("int", 3, 8),
        "rank": ("int", 1, 4),
        "memory": ("cat", [4, 8, 16, 32]),  # time-filter taps, first half of the layers
        "memory_late": ("cat", [8, 16, 32, 64]),  # second half (Google: 8 then 32)
        "bottleneck": ("cat", [0.33, 0.5, 1.0]),
    },
    # Attention: newer than the Apple/Google first stages. Causal, windowed, KV cache in PSRAM.
    "transformer": {
        "layers": ("int", 1, 6),
        "heads": ("cat", [1, 2, 4]),
        "window": ("cat", [8, 16, 32, 64]),  # tokens of look-back (x stride x 16 ms)
        "ffn_mult": ("cat", [1, 2, 4]),
        "pos": ("cat", ["rel_bias", "alibi", "rope", "none"]),
        "attn": ("cat", ["softmax", "sigmoid", "relu_linear"]),  # sigmoid/linear: no softmax unit
        "norm": ("cat", ["layernorm", "rmsnorm", "none"]),
        "stem": ("cat", ["linear", "conv3", "conv3x2"]),  # conv stem before attention
        "conv_module": ("cat", [False, True]),  # Conformer-lite: depthwise conv in each layer
    },
    # Hybrids: conv front end for local pattern, recurrent or attention back end for context.
    "crnn": {
        "conv_layers": ("int", 1, 3),
        "conv_kernel": ("cat", [3, 5]),
        "back": ("cat", ["gru", "lru", "attention"]),
        "back_layers": ("int", 1, 2),
    },
}
# A2 searches these; dnn and svdf get one baseline trial per size (weak or superseded in
# the earlier ladder), lstm/mamba_lite are left to A5 if their family wins.
STAGE_A_FAMILIES = ("tcn", "bcresnet", "rnn", "ssm", "transformer", "crnn")

DISTILL = {
    # --- teacher / targets ---------------------------------------------------------------
    "teacher": ("cat", ["speech_embedding", "whisper_tiny_enc"]),  # Google's (openWakeWord
    # backbone, 96-d / 80 ms) first; Whisper-tiny encoder (384-d / 20 ms, pooled) if A0-A3 stall
    "teacher_view": ("cat", ["same", "clean"]),  # teacher hears the same mix as the student, or
    # the clean close-talk source of it (simulator renders both): "clean" teaches the student
    # to output what the teacher would say without the noise/distance (denoising distillation)
    "target_norm": ("cat", ["none", "standardize", "whiten"]),  # per-dim scaling of the targets
    # --- loss ----------------------------------------------------------------------------
    "loss_cos": ("float", 0.0, 1.0),  # weight of (1 - cosine); MSE gets 1 - loss_cos
    "speech_weight": ("float", 1.0, 4.0),  # up-weight speech-bearing windows (teacher-energy gate)
    # --- data ----------------------------------------------------------------------------
    "audio_hours": ("cat", [25, 50, 100, 200]),  # simulated audio with teacher targets (~3 min
    # CPU per 25 h; ~100 MB per 25 h on disk)
    "speech_share": ("float", 0.5, 0.9),  # conversation vs ambience-only windows
    # --- optimiser -----------------------------------------------------------------------
    "optimizer": ("cat", ["adamw", "lion"]),
    "lr": ("log", 3e-4, 1e-2),  # lion uses lr/5
    "weight_decay": ("log", 1e-4, 1e-1),
    "warmup": ("cat", [100, 300, 1000]),
    "schedule": ("cat", ["cosine", "onecycle", "constant"]),
    "batch": ("cat", [256, 384, 512, 1024]),
    "grad_clip": ("cat", [0.5, 1.0, 5.0]),
    "ema": ("cat", [0.0, 0.999]),  # evaluate an exponential moving average of the weights
    "dropout": ("float", 0.0, 0.2),
    # --- augmentation (student input only; teacher targets stay fixed) ------------------
    "aug_gain_db": ("float", 0.0, 12.0),  # whole-window level shift (level robustness)
    "aug_tilt": ("float", 0.0, 1.0),  # spectral tilt (mic / distance colouring)
    "aug_band_mask_p": ("float", 0.0, 0.4),
    # --- quantisation --------------------------------------------------------------------
    "qat_start": ("float", 0.3, 0.8),  # fraction of training before fake-quant (4-bit only)
}
# Phase 06_probes defaults (what produced 0.700-0.788): teacher speech_embedding, view same,
# no target norm, MSE + cosine, ~75 h of windows, adamw, batch 256-384, 10000 steps.

STAGE_A_EVAL = {
    # Every trial reports all of these; selection uses "real_val".
    "sim_val": "multi_test (held-out voices, simtest mixer)",
    "real_val": "AMI far-field meetings (first half of meetings) + LibriSpeech dev-clean + "
                "MUSAN speech (half), through the chip front end; teacher on the same audio",
    "real_test": "AMI (other half of meetings, not the real_ami FA stream) + LibriSpeech "
                 "test-clean + MUSAN speech (other half); read once per finalist",
    "steps": "every 500 steps; prune at 3000 if not in the top third",
    "also_logged": "params, flip-flops (front end), QSPI bytes/step, on-chip bytes, receptive field",
}

# ======================================================================================
# 4. Stage B: DETECTOR (frozen student -> wake word). Not run before tapeout.
# ======================================================================================
DETECTOR = {
    # --- head (must fit next to the student: <10% of the budget) ------------------------
    "head_context": ("cat", [4, 6, 10, 16]),  # student outputs the head sees (x stride x 16 ms)
    "head_kind": ("cat", ["mlp", "conv1d", "gru"]),
    "head_hidden": ("cat", [0, 8, 16, 32, 64]),  # 0 = linear readout; 06_probes used 64 (66k
    # params, too big for the chip next to a 45k student)
    "head_dropout": ("float", 0.0, 0.4),  # 0.3 best in 06_probes
    "student_tune": ("cat", ["frozen", "last_block", "all"]),  # "all" is Stage C territory
    "smooth": ("int", 1, 8),  # moving average over the last n outputs before the threshold
    # (Google [G1] and microWakeWord use this); free at inference
    # --- optimiser -----------------------------------------------------------------------
    "lr": ("log", 1e-4, 3e-3),  # 3e-4 best in 06_probes
    "weight_decay": ("log", 1e-3, 3e-1),  # 0.1 best
    "steps": ("cat", [1500, 3000, 6000]),
    "batch": ("cat", [256, 512, 1024]),
    "seed": ("cat", [0, 1, 2]),  # heads are cheap: average seeds on every trial
    # --- loss ----------------------------------------------------------------------------
    "pos_frac": ("float", 0.2, 0.6),  # share of keyword windows per batch
    "pos_pool": ("cat", ["max", "logsumexp", "mean_top3"]),  # how a window's steps become one score
    "pos_last": ("int", 2, 12),  # student steps after the keyword end allowed to fire
    "neg_frame_weight": ("float", 0.0, 1.0),  # every-step negative BCE
    "neg_max_weight": ("float", 0.5, 2.0),  # max-over-window negative BCE
    "focal_gamma": ("float", 0.0, 2.0),
    "label_smooth": ("float", 0.0, 0.1),
    # --- negatives (the 1 FA/h gap is here: research/06_probes, conditions table) --------
    "neg_windows": ("cat", [25_000, 100_000, 400_000, 1_600_000]),  # conversation windows with
    # student embeddings; 06_probes had ~25k vs openWakeWord's thousands of hours
    "w_convo": ("float", 0.05, 1.0),
    "w_near": ("float", 0.05, 1.0),  # near-miss words ("telescope" vs "microscope")
    "w_other": ("float", 0.05, 1.0),  # the other candidate words
    "w_partial": ("float", 0.05, 1.0),  # keyword cut in half
    "w_amb": ("float", 0.05, 1.0),  # ambience only
    "mine_every": ("cat", [250, 500, 1000, 0]),  # 0 = no hard-negative mining
    "mine_frac": ("float", 0.1, 0.5),
    "mine_keep": ("cat", [1024, 4096, 16384]),
    # --- augmentation (on student embeddings) -------------------------------------------
    "aug_noise": ("float", 0.0, 0.2),  # gaussian on embeddings (0.05 in 06_probes)
    "aug_time_shift": ("int", 0, 3),  # steps the keyword end can move
    "aug_mixup_p": ("float", 0.0, 0.3),
}
# Fixed for Stage B: word=telescope (then the candidate list), metric = mean FRR at 1/2/5 FA/h
# on the simulated val half, ~2,000 test keywords (06_probes had 297: +-5-7 pts), reported by
# condition (good: <=1.2 m, open, SNR>=10 dB / OK: <=2.5 m, open, SNR>=0 / all).


def _suggest(trial, name, spec):
    kind = spec[0]
    if kind == "cat":
        return trial.suggest_categorical(name, spec[1])
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2], step=spec[3] if len(spec) > 3 else 1)
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    if kind == "log":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    raise ValueError(kind)


def sample_student(trial, size, families=STAGE_A_FAMILIES, frontend=None, engine=None, search_distill=False):
    """Stage A trial -> flat config. `size` is a SIZES constant (not searched). Pass frontend/
    engine dicts to hold them fixed (A2, A3), or None to search them (A1, A4). DISTILL knobs
    are searched only in A5; otherwise they take the 06_probes defaults (None here)."""
    cfg = {"stage": "A", "size": size, "family": trial.suggest_categorical("family", list(families))}
    for space, fixed in ((FRONTEND, frontend), (ENGINE, engine)):
        for k, spec in space.items():
            cfg[k] = fixed[k] if fixed is not None else _suggest(trial, k, spec)
    for k, spec in STUDENT_COMMON.items():
        cfg[k] = _suggest(trial, k, spec)
    for k, spec in FAMILIES[cfg["family"]].items():
        cfg[f"{cfg['family']}.{k}"] = _suggest(trial, f"{cfg['family']}.{k}", spec)
    for k, spec in DISTILL.items():
        cfg[k] = _suggest(trial, k, spec) if search_distill else None
    cfg["max_params"] = budget(cfg["weight_bits"], cfg["stride"])
    cfg["fits"] = size <= cfg["max_params"]
    return cfg


def sample_detector(trial, student_run):
    """Stage B trial -> flat config, on top of a finished Stage A run (its id)."""
    cfg = {"stage": "B", "student": student_run}
    for k, spec in DETECTOR.items():
        cfg[k] = _suggest(trial, k, spec)
    return cfg


def _n(spec):
    return len(spec[1]) if spec[0] == "cat" else (spec[2] - spec[1] + 1 if spec[0] == "int" else 10)


def size_of_space():
    """Rough count of distinct configs (categoricals x 10 per continuous)."""
    a = math.prod(_n(s) for s in {**FRONTEND, **ENGINE, **STUDENT_COMMON, **DISTILL}.values())
    return {"A:" + f: a * math.prod(_n(s) for s in F.values()) for f, F in FAMILIES.items()} | {
        "B": math.prod(_n(s) for s in DETECTOR.values())}


if __name__ == "__main__":
    for ws in (8, 4):
        print(f"{ws}-bit weights: max params by stride", {s: budget(ws, s) for s in (1, 2, 4)})
    print(f"on-chip working set at {TILES} tiles: {ON_CHIP_BYTES:.0f} B")
    print("sizes", SIZES, "fit at defaults:", [s <= budget(8, 2) for s in SIZES])
    for k, v in size_of_space().items():
        print(f"{k:16s} ~{v:.1e} configs")
