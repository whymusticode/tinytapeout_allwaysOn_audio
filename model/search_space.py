"""The full search space for the architecture + training study (Optuna), for review.

Read this file top to bottom: every searched knob is listed once, with its range and a
one-line reason. Nothing here runs training; `sample(trial)` turns an Optuna trial into a
config dict, and `budget()` is the hard 16-tile constraint every config must meet.

How the study runs (planned, model/tune.py):
  * Sampler: Optuna TPE, multivariate + group (handles the conditional per-family knobs).
  * Pruner: Hyperband on the existing validation (every 250 steps). A trial reports val_score
    (mean FRR at 0.5/1/2 FA/h on the simulated val half) and is stopped early if it is in the
    bottom 2/3 of trials at the same step. Min resource 750 steps, max 8000.
  * Objective: val_score of the best checkpoint. Simulated test half and real data are
    recorded for every trial but never seen by the sampler.
  * Size: every trial is sized to the 16-tile hardware budget (`budget()`), not searched as a
    free parameter. `budget_frac` sets how much of it a trial uses, so the study also learns
    whether filling the budget helps.
  * Word: one fixed word for the whole study (`telescope`, 3 syllables, mid-pack), so
    architecture/training choices aren't confounded with word difficulty.
  * Order: phase 0 probes (below) first, then ~200 trials of this space, then the top ~5
    configs re-run with 3 seeds and scored on the simulated test half.

Phase 0 probes (cheap, answer "where is the bottleneck?" before spending 200 trials):
  P1 test-set difficulty: run the public openWakeWord and microWakeWord "hey jarvis" models on
     our simulator's "hey jarvis" clips + our negative streams. If they also miss >50% at
     1 FA/h, our test set is harder than theirs; if they get <10%, our models/features are
     the problem.
  P2 capacity ceiling: one big unconstrained model (~2M params, no budget) on our features.
     Earlier ladder (arch_v1_recipe1): TCN 2k -> 295k params only moved sim FRR@1FA/h
     0.68 -> 0.56, which says capacity is not the main bottleneck.
  P3 front-end ceiling: same model on the same mixes with the front end varied one factor at
     a time: bands 16/24/32/40, hop 16/10 ms, uint8 vs float. If it is much better, the front
     end needs work, not the network. Area (yosys, our IIR filterbank, 68 FF per band):
     16 bands 1,269 FF (~4 tiles), 24: 1,813, 32: 2,357, 40: 2,901 FF (~9 tiles). The hop costs
     no area, but 10 ms frames leave 41 kB of QSPI per frame instead of 65 kB.
  P4 data: the first run on the v3 data (30 voices) at the old best config, versus v2.
  P5 teacher ceiling: frozen Google speech embedding (openWakeWord's backbone) + small head
     trained on our simulated data: what a pretrained speech representation buys.

Pretrained nets: see research/SOA.md section 10. Short version: none of them take our 16-band
16 ms features, so copying early layers directly doesn't apply; they enter as (a) teachers for
distillation (`distill_*` knobs), (b) the P1/P5 probes, (c) architecture templates
(BC-ResNet, MixConv). Our own pretraining on simulated speech (`pretrain`) is the in-feature-
space version of "start from a pretrained net".
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
    PSRAM state traffic is subtracted per model by the builder (it knows its state size)."""
    qspi = QSPI_BYTES_PER_FRAME * stride * 8 / weight_bits
    macs = MACS_PER_FRAME * stride
    return int(min(qspi, macs))


# ======================================================================================
# 2. The space
# ======================================================================================
# ("cat", [...])           categorical
# ("int", lo, hi[, step])  integer
# ("float", lo, hi)        uniform
# ("log", lo, hi)          log-uniform
# Family-specific knobs are only sampled for that family.

FRONTEND = {
    # The chip's filterbank (model/frontend_var.py, research/06_probes F2). Each choice costs area,
    # so it is scored against flip-flops, not just accuracy. lean5 bit widths are fixed (free cuts).
    "bands": ("cat", [8, 10, 12, 14, 16, 20, 24]),  # mel-spaced bands; ~68 FF/band at 2nd order
    # (lean5: ~55 FF/band, 4th order ~90); 8 -> -4.4 pts, 24 -> +1.1, 32 -> +1.8 on the 66-word probe
    "order": ("cat", [2, 4]),  # 4 = two cascaded resonators per band: +2.8 pts, +~35 FF/band
    "hop_ms": ("cat", [16, 32]),  # 16 vs 32 ms: -0.6 pts at 32 ms, halves the network's input rate
}

COMMON = {
    # --- size ----------------------------------------------------------------------------
    "budget_frac": ("float", 0.3, 1.0),  # fraction of budget() params to use; width is solved for it
    "weight_bits": ("cat", [8, 4]),  # 4-bit = QAT, doubles the params the bus allows
    # --- input ---------------------------------------------------------------------------
    "in_norm": ("cat", ["global", "running_mean", "pcen_lite"]),  # running per-band mean in the log
    # domain (cheap on chip): level/channel robustness, the classic far-field fix (PCEN)
    "in_norm_tau_s": ("log", 0.3, 5.0),  # time constant for running_mean / pcen_lite
    "in_deltas": ("cat", [False, True]),  # append frame-to-frame differences (16 more inputs)
    "stride": ("cat", [1, 2, 4]),  # output every 16/32/64 ms (frames stacked at the input)
    # --- head / post-processing ---------------------------------------------------------
    "head_hidden": ("cat", [0, 16, 32]),  # 0 = linear readout
    "smooth": ("int", 1, 8),  # moving average over the last n outputs before the threshold
    # (Google [G1] and microWakeWord use this); free at inference, tuned with the model
}

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
        "state_in_psram": ("cat", [False, True]),  # frees on-chip bytes for width, costs bus
        "prefix_windows": ("int", 0, 2),  # stateful training: preceding negative windows
    },
    # Linear recurrences trained in parallel, run as a recurrence: S4D / LRU / Mamba-lite.
    # minGRU (one of these) failed earlier only with per-frame norm; retried with in_norm options.
    "ssm": {
        "kind": ("cat", ["lru", "s4d", "mingru", "mamba_lite"]),
        "layers": ("int", 1, 4),
        "state": ("cat", [8, 16, 32, 64]),  # per-channel state size (lru/s4d/mamba)
        "expand": ("cat", [1, 2]),
        "conv_k": ("cat", [0, 3, 4]),  # short causal conv before the scan (Mamba-style)
        "prefix_windows": ("int", 0, 2),
    },
    # Dilated causal conv stack: TC-ResNet / microWakeWord MixedNet / WaveNet family.
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
        "prefix_windows": ("int", 0, 2),
    },
}

TRAIN = {
    # --- optimiser -----------------------------------------------------------------------
    "optimizer": ("cat", ["adamw", "lion"]),
    "lr": ("log", 3e-4, 1e-2),  # lion uses lr/5
    "weight_decay": ("log", 1e-4, 1e-1),
    "beta2": ("cat", [0.98, 0.999]),
    "warmup": ("cat", [100, 300, 1000]),
    "schedule": ("cat", ["cosine", "onecycle", "constant"]),
    "batch": ("cat", [256, 512, 1024, 2048]),
    "grad_clip": ("cat", [0.5, 1.0, 5.0]),
    "ema": ("cat", [0.0, 0.999]),  # evaluate an exponential moving average of the weights
    "dropout": ("float", 0.0, 0.3),
    # --- loss ----------------------------------------------------------------------------
    "pos_frac": ("float", 0.2, 0.6),  # share of keyword windows per batch (now 0.5)
    "pos_pool": ("cat", ["max", "logsumexp", "mean_top3"]),  # how a window's frames become one score
    "pos_last": ("int", 6, 40),  # frames after the keyword end allowed to fire (now 14)
    "neg_frame_weight": ("float", 0.0, 1.0),  # every-frame negative BCE (now 1)
    "neg_max_weight": ("float", 0.5, 2.0),  # max-over-window negative BCE (now 1)
    "focal_gamma": ("float", 0.0, 2.0),
    "label_smooth": ("float", 0.0, 0.1),
    # --- negatives -----------------------------------------------------------------------
    # relative weights of the negative groups (normalised); now .35/.10/.25/.10/.20
    "w_convo": ("float", 0.05, 1.0),
    "w_near": ("float", 0.05, 1.0),  # near-miss words ("telescope" vs "microscope")
    "w_other": ("float", 0.05, 1.0),  # the other 65 candidate words
    "w_partial": ("float", 0.05, 1.0),  # keyword cut in half
    "w_amb": ("float", 0.05, 1.0),  # ambience only
    "mine_every": ("cat", [250, 500, 1000, 0]),  # 0 = no hard-negative mining
    "mine_frac": ("float", 0.1, 0.5),
    "mine_keep": ("cat", [1024, 4096, 16384]),
    # --- augmentation (GPU, on features) -------------------------------------------------
    "aug_tilt": ("float", 0.0, 1.0),  # spectral tilt (mic / distance colouring)
    "aug_jitter": ("float", 0.0, 0.5),  # per-band gain jitter
    "aug_gain_db": ("float", 0.0, 12.0),  # whole-window level shift (level robustness)
    "aug_band_mask_p": ("float", 0.0, 0.6),
    "aug_band_mask_n": ("int", 1, 3),
    "aug_time_mask_p": ("float", 0.0, 0.5),  # SpecAugment time mask (up to 8 frames)
    "aug_time_shift": ("int", 0, 8),  # frames the keyword end can move
    "aug_mixup_p": ("float", 0.0, 0.3),  # mix a negative window under a positive (feature domain)
    # --- quantisation --------------------------------------------------------------------
    "qat_start": ("float", 0.3, 0.8),  # fraction of training before fake-quant turns on (4-bit)
    # --- pretraining / distillation (phase 2; needs teacher targets precomputed) ---------
    "pretrain": ("cat", ["none", "multiword", "phone_ctc"]),  # our own pretraining on simulated
    # speech: classify all 66 candidate words, or phone posteriors from Piper's phonemes (free
    # labels), then fine-tune on the target word. Apple's first stage is a phonetic model.
    "distill": ("cat", ["none", "speech_embedding", "whisper_tiny"]),  # teacher outputs on our
    # simulated audio; the student never sees real speech, only the teacher's predictions
    "distill_weight": ("float", 0.1, 1.0),
    "seed": ("cat", [0]),  # seeds are repeated only for the finalists
}

# Fixed for the whole study (not searched): word=telescope, data cache v3, frames 16 ms x 16
# bands uint8 (the chip front end), window 1.6 s, VAL_EVERY 250, patience 8, max 8000 steps,
# metric = mean FRR at 0.5/1/2 FA/h on the simulated val half.

# Phase 1 searches COMMON + FAMILIES + TRAIN except pretrain/distill (fixed to "none").
# Phase 2 adds them for the top families once teacher targets and pretraining runs exist.
PHASE1_FIXED = {"pretrain": "none", "distill": "none", "distill_weight": 0.0, "seed": 0}


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


def sample(trial, phase=1, families=None):
    """Optuna trial -> flat config dict. Family knobs are prefixed with the family name."""
    fams = families or list(FAMILIES)
    cfg = {"family": trial.suggest_categorical("family", fams)}
    for k, spec in {**FRONTEND, **COMMON}.items():
        cfg[k] = _suggest(trial, k, spec)
    for k, spec in FAMILIES[cfg["family"]].items():
        cfg[f"{cfg['family']}.{k}"] = _suggest(trial, f"{cfg['family']}.{k}", spec)
    for k, spec in TRAIN.items():
        if phase == 1 and k in PHASE1_FIXED:
            cfg[k] = PHASE1_FIXED[k]
        else:
            cfg[k] = _suggest(trial, k, spec)
    cfg["max_params"] = int(cfg["budget_frac"] * budget(cfg["weight_bits"], cfg["stride"]))
    return cfg


def size_of_space():
    """Rough count of distinct configs (categoricals x 10 per continuous) per family."""
    def n(spec):
        return len(spec[1]) if spec[0] == "cat" else (spec[2] - spec[1] + 1 if spec[0] == "int" else 10)
    base = math.prod(n(s) for s in {**FRONTEND, **COMMON}.values()) * math.prod(
        n(s) for k, s in TRAIN.items() if k not in PHASE1_FIXED)
    return {f: base * math.prod(n(s) for s in F.values()) for f, F in FAMILIES.items()}


if __name__ == "__main__":
    for ws in (8, 4):
        print(f"{ws}-bit weights: max params by stride", {s: budget(ws, s) for s in (1, 2, 4)})
    print(f"on-chip working set at {TILES} tiles: {ON_CHIP_BYTES:.0f} B")
    n_knobs = len(FRONTEND) + len(COMMON) + len(TRAIN) - len(PHASE1_FIXED)
    for f, F in FAMILIES.items():
        print(f"{f:12s} {n_knobs + len(F):3d} knobs, ~{size_of_space()[f]:.1e} configs")
