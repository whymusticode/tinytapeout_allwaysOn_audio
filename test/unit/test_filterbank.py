# SPDX-FileCopyrightText: © 2026 Michael Benton
# SPDX-License-Identifier: Apache-2.0
"""src/filterbank.v vs model/frontend_var.py (_fb_one at the chip's LEAN5 widths, then the
code_range clipping), bit for bit.

The Makefile runs it at the chip's setting and at the worst case for the cycle budget:
  FB_BANDS=12 FB_ORDER=4 (default)   FB_BANDS=24 FB_ORDER=4   FB_BANDS=16 FB_ORDER=2
Samples arrive every 1024 clocks, as in the chip (16.384 MHz / 16 kHz), so a too-slow
sequencer fails the busy check.
"""

import os
import sys

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "model"))
import frontend_var as FV  # noqa: E402

SAMPLE_CYCLES = 1024  # 16.384 MHz / 16 kHz
BANDS = int(os.environ.get("FB_BANDS", "12"))
ORDER = int(os.environ.get("FB_ORDER", "4"))
HOP = 256
V = FV.variant(**FV.LEAN5, bands=BANDS, order=ORDER, hop=HOP)


def reference(pcm):
    drop = 16 - V["pcm_bits"]
    x = ((pcm.astype(np.int64) >> drop) << drop)[None]
    g, a1, a2 = FV.coeffs_int(V)
    out = np.zeros((1, len(pcm) // HOP, BANDS), np.int64)
    FV._fb_batch(x, g, a1, a2, HOP, V["coef_frac"], V["y_w"], V["guard"], V["acc_w"], V["acc_shift"],
                 V["log_frac"], ORDER == 4, out)
    lo, levels = FV.code_range(V)
    return np.clip(out[0] - lo, 0, levels - 1)


def stimulus(n_frames, seed=0):
    """Quiet noise, speech-like bursts, and full-scale tones that saturate y and e."""
    rng = np.random.default_rng(seed)
    n = n_frames * HOP
    t = np.arange(n) / 16000
    x = 300 * rng.standard_normal(n) * (1 + np.sin(2 * np.pi * 4 * t))
    x[:HOP] = rng.integers(-3, 4, HOP)  # near-silent first frame
    k = 2 * HOP
    x[k : 2 * k] = 32767 * np.sin(2 * np.pi * 3000 * t[k : 2 * k])
    x[2 * k : 3 * k] = 32767 * np.sign(np.sin(2 * np.pi * 150 * t[2 * k : 3 * k]))
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


async def collect(dut, n_frames, out):
    for _ in range(n_frames):
        frame = np.zeros(BANDS, np.int64)
        await RisingEdge(dut.feat_valid)
        for b in range(BANDS):
            await FallingEdge(dut.clk)
            assert int(dut.feat_valid.value) == 1 and int(dut.feat_band.value) == b
            frame[b] = int(dut.feat.value)
        out.append(frame)


@cocotb.test()
async def test_filterbank_bit_exact(dut):
    n_frames = 8
    pcm = stimulus(n_frames)
    want = reference(pcm)
    assert want.max() == FV.code_range(V)[1] - 1, "stimulus should reach the top clip"
    assert ORDER == 2 or want.min() == 0, "stimulus should reach the bottom clip"  # order 2 leaks more

    cocotb.start_soon(Clock(dut.clk, 61, unit="ns").start())
    dut.pcm_valid.value = 0
    dut.pcm.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    got = []
    cocotb.start_soon(collect(dut, n_frames, got))
    for s in pcm:
        await FallingEdge(dut.clk)
        assert int(dut.busy.value) == 0, "previous sample not finished within 1024 clocks"
        dut.pcm.value = int(s) & 0xFFFF
        dut.pcm_valid.value = 1
        await FallingEdge(dut.clk)
        dut.pcm_valid.value = 0
        await ClockCycles(dut.clk, SAMPLE_CYCLES - 2)
    await ClockCycles(dut.clk, SAMPLE_CYCLES)
    got = np.array(got)
    assert got.shape == want.shape, (got.shape, want.shape)
    for f in range(n_frames):
        dut._log.info(f"frame {f}: rtl {got[f].tolist()}")
        assert np.array_equal(got[f], want[f]), f"frame {f}: ref {want[f].tolist()}"
