"""Fast, vectorized front end for training: the same integer arithmetic as the golden
model (frontend.filterbank), compiled with numba and run in parallel over clips.

    float audio (16 kHz) --pcm_from_float--> int16 PCM --filterbank--> uint8 codes

pcm_from_float stands in for mic + PDM + CIC: it applies the CIC's passband droop
and quantizes to int16. That step is a model of the analog world, not of the chip;
the CIC itself is checked bit-exact from PDM bits in the cocotb tests. Everything
after the int16 PCM is bit-exact with frontend.filterbank() and src/filterbank.v
(model/test_bitexact.py checks this).

Feature codes are log2 band energy in 1/8 octave steps (0..191). Divide by 8 for log2 units.
"""

import numba
import numpy as np

from frontend import (ACC_SHIFT, ACC_W, COEF_FRAC, GUARD, HOP, LOG_FRAC, N_BANDS, Y_W,
                      filterbank_coeffs)

RATE = 16000
CIC_ORDER, CIC_R, PDM_RATE = 3, 64, 1_024_000

_G, _A1, _A2 = filterbank_coeffs()


def cic_droop(x):
    """Apply the 3rd-order CIC (R=64 at 1.024 MHz) passband response, seen at 16 kHz."""
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(max(n, 2))))
    f = np.fft.rfftfreq(nfft, 1 / RATE)
    w = np.pi * f / PDM_RATE
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(w == 0, 1.0, np.sin(CIC_R * w) / (CIC_R * np.sin(w)))
    return np.fft.irfft(np.fft.rfft(x, nfft) * h**CIC_ORDER, nfft)[:n]


def pcm_from_float(x):
    """Float audio in [-1, 1] at 16 kHz -> int16 PCM as the CIC would output it."""
    y = cic_droop(np.asarray(x, dtype=np.float64)) * 32768
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


@numba.njit(cache=True, inline="always")
def _log2_code(acc):
    if acc == 0:
        return 0
    p = 63
    while (acc >> p) == 0:
        p -= 1
    m = ((acc << LOG_FRAC) >> p) & ((1 << LOG_FRAC) - 1)
    return p * (1 << LOG_FRAC) + m


@numba.njit(cache=True)
def _filterbank_one(pcm, g, a1, a2, out):
    n_frames = out.shape[0]
    y_hi = (1 << (Y_W - 1)) - 1
    y_lo = -y_hi - 1
    acc_max = (1 << ACC_W) - 1
    rnd = 1 << (COEF_FRAC - 1)
    y1 = np.zeros(N_BANDS, np.int64)
    y2 = np.zeros(N_BANDS, np.int64)
    acc = np.zeros(N_BANDS, np.int64)
    xm1 = 0
    xm2 = 0
    for n in range(n_frames * HOP):
        x = np.int64(pcm[n])
        v = (x - xm2) << GUARD
        xm2 = xm1
        xm1 = x
        for b in range(N_BANDS):
            y = (g[b] * v - a1[b] * y1[b] - a2[b] * y2[b] + rnd) >> COEF_FRAC
            y = min(max(y, y_lo), y_hi)
            y2[b] = y1[b]
            y1[b] = y
            acc[b] = min(acc_max, acc[b] + (abs(y) >> ACC_SHIFT))
        if n % HOP == HOP - 1:
            f = n // HOP
            for b in range(N_BANDS):
                out[f, b] = _log2_code(acc[b])
                acc[b] = 0


@numba.njit(cache=True, parallel=True)
def _filterbank_batch(pcm, g, a1, a2, out):
    for i in numba.prange(pcm.shape[0]):
        _filterbank_one(pcm[i], g, a1, a2, out[i])


def filterbank(pcm):
    """int16 PCM, shape (n,) or (batch, n) -> uint8 codes (..., n // HOP, N_BANDS).
    Batches run in parallel over clips (numba threads; set NUMBA_NUM_THREADS to limit)."""
    pcm = np.ascontiguousarray(pcm, dtype=np.int16)
    single = pcm.ndim == 1
    if single:
        pcm = pcm[None]
    out = np.zeros((pcm.shape[0], pcm.shape[1] // HOP, N_BANDS), np.uint8)
    if single:
        _filterbank_one(pcm[0], _G, _A1, _A2, out[0])
        return out[0]
    _filterbank_batch(pcm, _G, _A1, _A2, out)
    return out


def features(x):
    """Float audio in [-1, 1] at 16 kHz -> (n_frames, N_BANDS) uint8 feature codes."""
    return filterbank(pcm_from_float(x))
