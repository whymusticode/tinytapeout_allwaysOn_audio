"""Workstation check (repo env): the Pi decoder matches the chip's golden model bit for bit.

  python -m pytest hw/pi_mics/test_pdm.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "model"))
import frontend  # noqa: E402
import pdm  # noqa: E402


def sigma_delta(x):
    """1st-order sigma-delta modulator: float in [-1, 1] -> 0/1 bits."""
    out = np.zeros(len(x), np.uint8)
    acc = 0.0
    for i, v in enumerate(x):
        acc += v
        b = acc >= 0
        out[i] = b
        acc -= 1.0 if b else -1.0
    return out


def fake_capture(bits0, bits1, phase0=0, phase1=1, rng=None):
    """Interleave two mics as pdm_capture.c would deliver them; the undriven phase reads noise."""
    rng = rng or np.random.default_rng(0)
    n = len(bits0)
    g = np.zeros((n, 2), np.uint8)
    noise = rng.integers(0, 2, (n, 2), dtype=np.uint8)
    for m, (b, p) in enumerate(((bits0, phase0), (bits1, phase1))):
        lane = noise[:, 1 - p] if m else np.zeros(n, np.uint8)  # mic0's off phase: pulled low
        g[:, p] |= b.astype(np.uint8) << m
        g[:, 1 - p] |= lane << m
    flat = g.reshape(-1, 16)  # 16 groups per word, MSB first
    w = np.zeros(len(flat), np.uint32)
    for j in range(16):
        w |= flat[:, j].astype(np.uint32) << np.uint32(30 - 2 * j)
    return w.astype("<u4").tobytes()


def test_cic_streaming_bitexact():
    rng = np.random.default_rng(1)
    t = np.arange(64 * 3000) / pdm.PDM_HZ
    bits = sigma_delta(0.6 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.normal(size=len(t)))
    ref = frontend.cic3(bits)
    c = pdm.Cic3()
    cuts = np.sort(rng.integers(0, len(bits), 20))  # uneven chunks, not multiples of 64
    got = np.concatenate([c.run(b) for b in np.split(bits, cuts)])
    assert np.array_equal(got, ref)


def test_unpack_and_phase():
    rng = np.random.default_rng(2)
    n = 64 * 1024
    t = np.arange(n) / pdm.PDM_HZ
    b0 = sigma_delta(0.3 * np.sin(2 * np.pi * 300 * t))
    b1 = sigma_delta(0.3 * np.sin(2 * np.pi * 1000 * t))
    s = pdm.split(fake_capture(b0, b1, phase0=0, phase1=1, rng=rng))
    assert pdm.find_phase(s[0]) == 0 and pdm.find_phase(s[1]) == 1
    assert np.array_equal(s[0, 0], b0) and np.array_equal(s[1, 1], b1)
    assert np.array_equal(pdm.Cic3().run(s[1, 1]), frontend.cic3(b1))
