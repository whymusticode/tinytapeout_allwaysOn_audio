"""Front-end variants for the bit-diet probes: the chip's filterbank arithmetic with every
width a parameter, plus float references (ideal 16-band IIR, log-mel).

A variant is a dict; FE_BASE is today's chip (src/filterbank.v, bit-exact with features.py
when every field is at its default). Each field maps to hardware:

  rate        PCM sample rate (16000 or 8000; 8 kHz halves the filter work, bands stop at 3.8 kHz)
  pcm_bits    bits kept of the int16 CIC output (LSBs dropped: quieter sounds sink in noise)
  bands       filterbank bands (mel-spaced, 150 Hz .. 0.44*rate)
  coef_frac   fractional bits of the biquad coefficients (Q2.coef_frac); multiplier cycles
  y_w         filter state width (2 registers per band)
  guard       left shift of the input difference before filtering (headroom vs precision)
  acc_w       energy accumulator width (1 register per band)
  acc_shift   right shift of |y| before accumulating
  log_frac    fractional bits of the log2 code (3: 1/8 octave)
  feat_bits   bits kept of the final code: codes are clipped to [floor, floor + 2^feat_bits),
              floor = log2 energy 7 (codes below are digital silence / far below the mic noise;
              measured code range of the probe data at 1/8 octave: 1st-99.9th pct 56-180).
              Pair fewer bits with a coarser log step (log_frac) to keep the range.
  hop         samples per frame (256 = 16 ms at 16 kHz)
  kind        "chip" (above), "float_iir" (same bands, float math, natural log energy),
              "mel" (STFT log-mel, n_mels = bands, 25 ms window)
"""

import math

import numba
import numpy as np
from scipy.signal import iirpeak, resample_poly

FE_BASE = dict(kind="chip", rate=16000, pcm_bits=16, bands=16, coef_frac=14, y_w=22, guard=4,
               acc_w=24, acc_shift=2, log_frac=3, feat_bits=8, hop=256,
               energy="abs", order=2, q_mult=1.0)  # float_iir only: |y| or y^2, 2 or 4 poles, Q scale
F_LO = 150.0
CIC_ORDER, CIC_R, PDM_RATE = 3, 64, 1_024_000


def variant(**kw):
    v = dict(FE_BASE)
    v.update(kw)
    return v


def frame_s(v):
    return v["hop"] / v["rate"]


def _mel(f):
    return 2595 * np.log10(1 + f / 700)


def _imel(m):
    return 700 * (10 ** (m / 2595) - 1)


def band_centers(v):
    hi = min(7000.0, 0.4375 * v["rate"])
    return _imel(np.linspace(_mel(F_LO), _mel(hi), v["bands"]))


def coeffs_float(v):
    """Same design as frontend.filterbank_float, for any band count / rate."""
    fc = band_centers(v)
    edges = _imel(np.linspace(_mel(F_LO), _mel(fc[-1]), v["bands"]) + np.diff(_mel(fc)).mean() / 2)
    bw = np.diff(np.concatenate([[F_LO * 0.8], edges]))
    out = []
    for f, b in zip(fc, bw):
        bn, an = iirpeak(f, Q=max(f / max(b, 1.0), 0.7) * v.get("q_mult", 1.0), fs=v["rate"])
        out.append((bn[0], an[1], an[2]))
    return np.array(out)


def coeffs_int(v):
    q = np.round(coeffs_float(v) * (1 << v["coef_frac"])).astype(np.int64)
    return q[:, 0].copy(), q[:, 1].copy(), q[:, 2].copy()


def cic_droop(x, rate):
    n = x.shape[-1]
    nfft = 1 << int(np.ceil(np.log2(max(n, 2))))
    f = np.fft.rfftfreq(nfft, 1 / 16000)
    w = np.pi * f / PDM_RATE
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(w == 0, 1.0, np.sin(CIC_R * w) / (CIC_R * np.sin(w)))
    return np.fft.irfft(np.fft.rfft(x, nfft, axis=-1) * h**CIC_ORDER, nfft, axis=-1)[..., :n]


def to_pcm(audio16, v):
    """int16 audio at 16 kHz (as the simulator wrote it) -> int64 PCM as this chip's CIC
    would deliver it: droop, optional 8 kHz, LSBs dropped."""
    x = cic_droop(audio16.astype(np.float64) / 32768, 16000)
    if v["rate"] != 16000:
        x = resample_poly(x, v["rate"], 16000, axis=-1)
    p = np.clip(np.round(x * 32768), -32768, 32767).astype(np.int64)
    drop = 16 - v["pcm_bits"]
    if drop > 0:
        p = (p >> drop) << drop  # truncate like a narrower CIC output (arithmetic shift)
    return p


@numba.njit(cache=True)
def _fb_one(pcm, g, a1, a2, hop, coef_frac, y_w, guard, acc_w, acc_shift, log_frac, order4, out):
    nb = g.shape[0]
    n_frames = out.shape[0]
    y_hi = (1 << (y_w - 1)) - 1
    y_lo = -y_hi - 1
    acc_max = (1 << acc_w) - 1
    rnd = 1 << (coef_frac - 1)
    y1 = np.zeros(nb, np.int64)
    y2 = np.zeros(nb, np.int64)
    z1 = np.zeros(nb, np.int64)  # order 4: second resonator, input y[n] - y[n-2] (= y - y2 before update)
    z2 = np.zeros(nb, np.int64)
    acc = np.zeros(nb, np.int64)
    xm1 = np.int64(0)
    xm2 = np.int64(0)
    for n in range(n_frames * hop):
        x = pcm[n]
        vv = (x - xm2) << guard
        xm2 = xm1
        xm1 = x
        for b in range(nb):
            y = (g[b] * vv - a1[b] * y1[b] - a2[b] * y2[b] + rnd) >> coef_frac
            y = min(max(y, y_lo), y_hi)
            w = y - y2[b]
            y2[b] = y1[b]
            y1[b] = y
            if order4:
                z = (g[b] * w - a1[b] * z1[b] - a2[b] * z2[b] + rnd) >> coef_frac
                z = min(max(z, y_lo), y_hi)
                z2[b] = z1[b]
                z1[b] = z
                y = z
            acc[b] = min(acc_max, acc[b] + (abs(y) >> acc_shift))
        if n % hop == hop - 1:
            f = n // hop
            for b in range(nb):
                a = acc[b]
                if a == 0:
                    out[f, b] = 0
                else:
                    p = 63
                    while (a >> p) == 0:
                        p -= 1
                    m = ((a << log_frac) >> p) & ((1 << log_frac) - 1)
                    out[f, b] = p * (1 << log_frac) + m
                acc[b] = 0


@numba.njit(cache=True, parallel=True)
def _fb_batch(pcm, g, a1, a2, hop, coef_frac, y_w, guard, acc_w, acc_shift, log_frac, order4, out):
    for i in numba.prange(pcm.shape[0]):
        _fb_one(pcm[i], g, a1, a2, hop, coef_frac, y_w, guard, acc_w, acc_shift, log_frac, order4, out[i])


def code_range(v):
    """(floor, n_levels) of the emitted code. The raw log2 code spans 0 .. acc_w*2^log_frac;
    feat_bits keeps 2^feat_bits levels from log2 energy 7 up (or the whole range if it fits)."""
    top = v["acc_w"] * (1 << v["log_frac"])
    levels = 1 << v["feat_bits"]
    if levels >= top:
        return 0, top
    return min(7 << v["log_frac"], top - levels), levels


@numba.njit(cache=True, parallel=True)
def _fb_float_batch(x, g, a1, a2, hop, power, order4, out):
    for i in numba.prange(x.shape[0]):
        nb = g.shape[0]
        y1 = np.zeros(nb)
        y2 = np.zeros(nb)
        z1 = np.zeros(nb)  # second section (order 4): same resonator again on y
        z2 = np.zeros(nb)
        ym1 = np.zeros(nb)
        ym2 = np.zeros(nb)
        acc = np.zeros(nb)
        xm1 = 0.0
        xm2 = 0.0
        for n in range(out.shape[1] * hop):
            xx = x[i, n]
            vv = xx - xm2
            xm2 = xm1
            xm1 = xx
            for b in range(nb):
                y = g[b] * vv - a1[b] * y1[b] - a2[b] * y2[b]
                y2[b] = y1[b]
                y1[b] = y
                if order4:
                    w = y - ym2[b]
                    ym2[b] = ym1[b]
                    ym1[b] = y
                    z = g[b] * w - a1[b] * z1[b] - a2[b] * z2[b]
                    z2[b] = z1[b]
                    z1[b] = z
                    y = z
                acc[b] += y * y if power else abs(y)
            if n % hop == hop - 1:
                for b in range(nb):
                    e = acc[b] / hop + 1e-12
                    out[i, n // hop, b] = 0.5 * math.log2(e) if power else math.log2(e)
                    acc[b] = 0.0


def features(audio16, v):
    """(N, n) int16 audio at 16 kHz -> (N, frames, bands) features. Chip variants return
    uint8/uint16 codes (the network sees code / 2^log_frac, i.e. log2 units); float variants
    return float32 log2 energies."""
    audio16 = np.atleast_2d(audio16)
    if v["kind"] == "mel":
        return mel(audio16, v)
    if v["kind"] == "float_iir":
        x = cic_droop(audio16.astype(np.float64) / 32768, 16000)
        if v["rate"] != 16000:
            x = resample_poly(x, v["rate"], 16000, axis=-1)
        g, a1, a2 = (np.ascontiguousarray(c) for c in coeffs_float(v).T)
        out = np.zeros((x.shape[0], x.shape[1] // v["hop"], v["bands"]))
        _fb_float_batch(np.ascontiguousarray(x), g, a1, a2, v["hop"], v.get("energy", "abs") == "pow",
                        v.get("order", 2) == 4, out)
        return out.astype(np.float32)
    pcm = np.ascontiguousarray(to_pcm(audio16, v))
    g, a1, a2 = coeffs_int(v)
    out = np.zeros((pcm.shape[0], pcm.shape[1] // v["hop"], v["bands"]), np.int64)
    _fb_batch(pcm, g, a1, a2, v["hop"], v["coef_frac"], v["y_w"], v["guard"], v["acc_w"],
              v["acc_shift"], v["log_frac"], v.get("order", 2) == 4, out)
    lo, levels = code_range(v)
    out = np.clip(out - lo, 0, levels - 1)
    return out.astype(np.uint8 if levels <= 256 else np.uint16)


def to_log2(f, v):
    """Features -> float log2 units for the network (comparable scale across variants)."""
    if v["kind"] != "chip":
        return f.astype(np.float32)
    lo, _ = code_range(v)
    return (f.astype(np.float32) + lo) / (1 << v["log_frac"])


def mel(audio16, v, win_s=0.025):
    import torch

    x = torch.from_numpy(cic_droop(audio16.astype(np.float64) / 32768, 16000)).float()
    rate = v["rate"]
    if rate != 16000:
        x = torch.from_numpy(resample_poly(x.numpy(), rate, 16000, axis=-1)).float()
    n_fft = 1 << int(np.ceil(np.log2(win_s * rate)))
    hop = v["hop"]
    # causal framing: frame j covers the window ending at sample (j+1)*hop
    xp = torch.nn.functional.pad(x, (n_fft - hop, 0))
    S = torch.stft(xp, n_fft, hop, win_length=int(win_s * rate), window=torch.hann_window(int(win_s * rate)),
                   center=False, return_complex=True).abs() ** 2
    f = np.linspace(0, rate / 2, n_fft // 2 + 1)
    hi = min(7000.0, 0.4375 * rate)
    pts = _imel(np.linspace(_mel(F_LO * 0.8), _mel(hi * 1.1), v["bands"] + 2))
    fb = np.zeros((v["bands"], len(f)))
    for b in range(v["bands"]):
        l, c, r = pts[b : b + 3]
        fb[b] = np.clip(np.minimum((f - l) / (c - l), (r - f) / (r - c)), 0, None)
    m = torch.einsum("bf,nft->ntb", torch.from_numpy(fb).float(), S)
    n_frames = x.shape[-1] // hop
    return (torch.log2(m + 1e-9)[:, :n_frames] / 2).numpy().astype(np.float32)  # /2: amplitude-log2 scale


# --- area model (flip-flops), calibrated on yosys runs of src/filterbank.v ------------------

def flipflops(v):
    """Filterbank flip-flops. Per band: 2 state registers (y_w) + accumulator (acc_w), stored in
    rotating shift registers. Shared: x[n-1], x[n-2] (pcm_bits), input difference (pcm_bits+1+
    guard), product accumulator, shifted operand (y_w+16+2 each), coefficient shifter (coef
    bits), counters. yosys: 16 bands default = 1,269 FF, 40 bands = 2,901 FF (68 per band)."""
    if v["kind"] != "chip":
        return None
    nb = v["bands"]
    coef_w = v["coef_frac"] + 2
    per_band = (4 if v.get("order", 2) == 4 else 2) * v["y_w"] + v["acc_w"]
    p_w = coef_w + v["y_w"] + 2
    shared = 2 * v["pcm_bits"] + (v["pcm_bits"] + 1 + v["guard"]) + 2 * p_w + coef_w
    counters = int(np.ceil(np.log2(v["hop"]))) + int(np.ceil(np.log2(max(nb, 2)))) + 2 + 4 + 2
    out = 8 + int(np.ceil(np.log2(max(nb, 2)))) + 1  # feat, feat_band, feat_valid
    return nb * per_band + shared + counters + out


def cycles_per_sample(v):
    """Bit-serial multiplier: 3 products x coefficient bits x bands (+2)."""
    return 3 * (2 if v.get("order", 2) == 4 else 1) * (v["coef_frac"] + 2) * v["bands"] + 2


def cycle_budget(v, clk=16.384e6):
    return clk / v["rate"]


def tiles(v):
    ff = flipflops(v)
    return None if ff is None else ff / 320
