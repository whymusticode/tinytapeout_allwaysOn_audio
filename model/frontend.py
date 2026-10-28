"""Bit-exact Python reference ("golden model") for the audio front end.

    src/pdm_cic.v     cic3()        PDM bits -> 16 kHz int16 PCM
    src/vad.v         vad()         PCM -> voice activity
    filterbank()                    PCM -> (frames, 16) uint8 log2 band energies: the first,
                                    2nd-order, full-width chip front end. src/filterbank.v now
                                    implements Stage A's (4th order, frontend_var.LEAN5 widths)
                                    and is checked against frontend_var._fb_one instead.

Written for clarity, one sample at a time, mirroring the RTL. Every function here
must match the RTL bit-for-bit (the cocotb tests compare against it). The fast,
vectorized version used for training is features.py; model/test_bitexact.py checks
that it matches this file exactly.
"""

import numpy as np
from scipy.signal import iirpeak

PDM_RATE = 1_024_000
DECIM = 64
PCM_RATE = PDM_RATE // DECIM  # 16 kHz
CIC_W = 25  # any R up to 255 (pdm_cic.v W); R = 64 alone needs 20
VAD_ENV_SHIFT = 6


def pdm_modulate(x, seed=0):
    """2nd-order delta-sigma modulator: float samples in [-1, 1] at PDM_RATE -> 0/1 bits.

    Stands in for a MEMS mic. Not part of the chip, so it need not be bit-exact
    with anything, only deterministic.
    """
    rng = np.random.default_rng(seed)
    dither = rng.uniform(-1e-3, 1e-3, len(x))
    bits = np.empty(len(x), dtype=np.uint8)
    s1 = s2 = 0.0
    y = 0.0
    for n, v in enumerate(x):
        s1 += v - y
        s2 += s1 - y
        b = s2 + dither[n] >= 0
        bits[n] = b
        y = 1.0 if b else -1.0
    return bits


def upsample_to_pdm(pcm_float, rate):
    """Resample float audio at `rate` to PDM_RATE (linear interpolation)."""
    n_out = int(round(len(pcm_float) * PDM_RATE / rate))
    t = np.arange(n_out) * (rate / PDM_RATE)
    return np.interp(t, np.arange(len(pcm_float)), pcm_float)


def _wrap(v, w):
    m = 1 << w
    v &= m - 1
    return v - m if v >= m >> 1 else v


def cic3(bits, decim=DECIM, shift=3):
    """3rd-order CIC, R = decim, on 0/1 PDM bits -> int16 PCM (matches pdm_cic.v; the defaults
    are the chip's power-up setting, 1.024 MHz -> 16 kHz). PCM = sat16(c3 >> shift)."""
    w = CIC_W
    i1 = i2 = i3 = 0
    i3_d = c1_d = c2_d = 0
    out = []
    for n, b in enumerate(bits):
        x = 1 if b else -1
        # Pipelined integrators: every stage uses the previous stage's old value.
        i1, i2, i3 = _wrap(i1 + x, w), _wrap(i2 + i1, w), _wrap(i3 + i2, w)
        if n % decim == decim - 1:
            c1 = _wrap(i3 - i3_d, w)
            c2 = _wrap(c1 - c1_d, w)
            c3 = _wrap(c2 - c2_d, w)
            i3_d, c1_d, c2_d = i3, c1, c2
            out.append(max(-32768, min(32767, c3 >> shift)))
    return np.array(out, dtype=np.int16)


def vad(pcm, threshold):
    """Energy-envelope VAD (matches vad.v). Returns (active[], envelope[]) per sample,
    sampled after each update, as the RTL registers them."""
    env = 0
    active = np.zeros(len(pcm), dtype=np.uint8)
    envelope = np.zeros(len(pcm), dtype=np.int32)
    for n, p in enumerate(pcm):
        mag = min(abs(int(p)), 32767)
        act = (env >> 4) > threshold
        env = env + (((mag << 4) - env) >> VAD_ENV_SHIFT)
        active[n] = act
        envelope[n] = env >> 4
    return active, envelope


# --- Filterbank: 16 mel-spaced band-pass biquads -> frame energy -> log2 ---
#
# Per band b, per PCM sample x[n] (int16):
#   v    = (x[n] - x[n-2]) << GUARD                   band-pass numerator (1 - z^-2), shared by all bands
#   s    = g*v - a1*y[n-1] - a2*y[n-2] + 2^(COEF_FRAC-1)
#   y[n] = sat_Y_W(s >>> COEF_FRAC)                   GUARD extra fraction bits keep the
#                                                      low bands (poles near z=1) quiet
#   acc  = sat_ACC_W(acc + (|y[n]| >> ACC_SHIFT))
# Every HOP samples: feature = log2_code(acc), acc = 0.
# log2_code: msb position p and the next LOG_FRAC bits m -> p * 2^LOG_FRAC + m (0 if acc == 0).
# Coefficients are Q2.14 signed 16-bit constants (src/filterbank_coeffs.vh, generated).

RATE = PCM_RATE
HOP = 256  # 16 ms frames
N_BANDS = 16
F_LO, F_HI = 150.0, 7000.0
COEF_FRAC = 14
GUARD = 4
Y_W = 22
ACC_SHIFT = 2
ACC_W = 24
LOG_FRAC = 3
FEAT_MAX = (ACC_W - 1) * (1 << LOG_FRAC) + (1 << LOG_FRAC) - 1  # 191


def _mel(f):
    return 2595 * np.log10(1 + f / 700)


def _imel(m):
    return 700 * (10 ** (m / 2595) - 1)


def band_centers():
    return _imel(np.linspace(_mel(F_LO), _mel(F_HI), N_BANDS))


def filterbank_float():
    """(g, a1, a2) float per band. Q set so neighbouring bands cross at about -3 dB."""
    fc = band_centers()
    edges = _imel(np.linspace(_mel(F_LO), _mel(F_HI), N_BANDS) + np.diff(_mel(fc)).mean() / 2)
    bw = np.diff(np.concatenate([[F_LO * 0.8], edges]))
    out = []
    for f, b in zip(fc, bw):
        bn, an = iirpeak(f, Q=max(f / max(b, 1.0), 0.7), fs=RATE)
        assert bn[1] == 0 and np.isclose(bn[2], -bn[0]) and an[0] == 1
        out.append((bn[0], an[1], an[2]))
    return np.array(out)


def filterbank_coeffs():
    """Quantized (g, a1, a2), each an int array of N_BANDS signed 16-bit values."""
    q = np.round(filterbank_float() * (1 << COEF_FRAC)).astype(np.int64)
    assert q.min() >= -(1 << 15) and q.max() < (1 << 15)
    return q[:, 0], q[:, 1], q[:, 2]


def _sat(v, w):
    hi = (1 << (w - 1)) - 1
    return max(-hi - 1, min(hi, v))


def log2_code(acc):
    if acc == 0:
        return 0
    p = acc.bit_length() - 1
    m = (acc << LOG_FRAC >> p) & ((1 << LOG_FRAC) - 1)
    return p * (1 << LOG_FRAC) + m


def filterbank(pcm):
    """int16 PCM -> (len(pcm) // HOP, N_BANDS) uint8 feature codes (matches filterbank.v)."""
    g, a1, a2 = (list(map(int, c)) for c in filterbank_coeffs())
    acc_max = (1 << ACC_W) - 1
    y1 = [0] * N_BANDS
    y2 = [0] * N_BANDS
    acc = [0] * N_BANDS
    xm1 = xm2 = 0
    out = []
    for n, x in enumerate(pcm):
        x = int(x)
        v = (x - xm2) << GUARD
        xm2, xm1 = xm1, x
        for b in range(N_BANDS):
            s = g[b] * v - a1[b] * y1[b] - a2[b] * y2[b] + (1 << (COEF_FRAC - 1))
            y = _sat(s >> COEF_FRAC, Y_W)
            y2[b], y1[b] = y1[b], y
            acc[b] = min(acc_max, acc[b] + (abs(y) >> ACC_SHIFT))
        if n % HOP == HOP - 1:
            out.append([log2_code(a) for a in acc])
            acc = [0] * N_BANDS
    return np.array(out, dtype=np.uint8).reshape(-1, N_BANDS)
