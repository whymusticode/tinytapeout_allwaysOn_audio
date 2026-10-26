"""Decode the raw PIO capture of two PDM mics into the chip's 16 kHz PCM (numpy only).

Capture format (pdm_capture.c): little-endian uint32 words, MSB first in time. Each word holds
16 two-bit groups; groups alternate between the two halves of the PDM clock:
  even group: sampled at the end of the LOW half  (just before the rising edge)
  odd group:  sampled at the end of the HIGH half (just before the falling edge)
bit 0 of a group = DAT0 (mic 0), bit 1 = DAT1 (mic 1).

A mic drives its data line during one half only (set by its SEL pin), so for each mic one
phase carries its PDM bits; find_phase() picks it from the signal statistics.

The chip's naming (pdm_cic.v, ui[1] PDM_EDGE): edge=1 is the low-half phase (even groups),
edge=0 is the high-half phase (odd groups).

Cic3 is a vectorised, streaming, bit-exact copy of model/frontend.py:cic3() (and pdm_cic.v),
so PCM recorded by the Pi is exactly what the chip computes from the same mic bits.
"""

import numpy as np

PDM_HZ = 1_024_000
DECIM = 64
RATE = PDM_HZ // DECIM
CIC_W = 20
MASK = (1 << CIC_W) - 1


def unpack(raw):
    """bytes from pdm_capture -> (bits[mic, phase, n] uint8), phase 0 = edge 1 (low half)."""
    w = np.frombuffer(raw, dtype="<u4")
    shifts = np.arange(30, -1, -2, dtype=np.uint32)
    g = ((w[:, None] >> shifts) & 3).astype(np.uint8).reshape(-1, 2)  # (n, phase)
    return np.stack([g & 1, g >> 1])  # (mic, n, phase)


def split(raw):
    b = unpack(raw)
    return b.transpose(0, 2, 1)  # (mic, phase, n)


def find_phase(bits_mic):
    """bits_mic: (2 phases, n) -> the phase the mic drives, or None.
    A PDM modulator pushes its quantisation noise up to high frequencies, so consecutive bits
    are anti-correlated (lag-1 correlation well below 0). An undriven phase reads the pull
    resistor (constant, correlation +1) or floating noise (correlation ~0)."""
    best, score = None, -0.05
    for p in (0, 1):
        x = bits_mic[p].astype(np.float32) * 2 - 1
        if len(x) < 2 or not 0.2 < (x > 0).mean() < 0.8:
            continue
        r = float(np.mean(x[1:] * x[:-1]) - np.mean(x) ** 2)
        if r < score:
            best, score = p, r
    return best


def phase_to_edge(p):
    return 1 if p == 0 else 0


def _wrap(x):
    """Two's-complement wrap to CIC_W bits, as signed int64."""
    x = x & MASK
    return np.where(x >= 1 << (CIC_W - 1), x - (1 << CIC_W), x)


class Cic3:
    """3rd-order CIC, R=64, pipelined integrators, 20-bit wrap, output >> 3, int16 saturation.
    Feed any number of bits per call; state carries over."""

    def __init__(self):
        self.i1 = self.i2 = self.i3 = 0
        self.d = [0, 0, 0]  # comb delays: i3, c1, c2 at the previous output
        self.n = 0  # bits consumed so far

    def run(self, bits):
        bits = np.asarray(bits)
        if len(bits) == 0:
            return np.zeros(0, np.int16)
        x = np.where(bits != 0, 1, -1).astype(np.int64)
        # i1[n] = i1 + sum x[..n]; i2[n] = i2 + sum i1[..n-1]; i3[n] = i3 + sum i2[..n-1]
        i1 = (self.i1 + np.cumsum(x)) & MASK
        i2 = (self.i2 + np.cumsum(np.concatenate(([self.i1], i1[:-1])))) & MASK
        i3 = (self.i3 + np.cumsum(np.concatenate(([self.i2], i2[:-1])))) & MASK
        self.i1, self.i2, self.i3 = int(i1[-1]), int(i2[-1]), int(i3[-1])
        idx = np.nonzero((self.n + np.arange(len(bits))) % DECIM == DECIM - 1)[0]
        self.n += len(bits)
        if len(idx) == 0:
            return np.zeros(0, np.int16)
        v = i3[idx]
        out = []
        for k in range(3):  # comb stages
            prev = np.concatenate(([self.d[k]], v[:-1]))
            self.d[k] = int(v[-1])
            v = (v - prev) & MASK
        y = _wrap(v) >> 3
        return np.clip(y, -32768, 32767).astype(np.int16)
