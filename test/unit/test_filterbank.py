# SPDX-FileCopyrightText: © 2026 Michael Benton
# SPDX-License-Identifier: Apache-2.0
"""src/filterbank.v vs model/frontend.py:filterbank(), bit for bit."""

import os
import sys

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "model"))
import frontend  # noqa: E402

SAMPLE_CYCLES = 1024  # 16.384 MHz / 16 kHz


def stimulus(n_frames, seed=0):
    """Quiet noise, speech-like bursts, and full-scale tones that saturate y and e."""
    rng = np.random.default_rng(seed)
    n = n_frames * frontend.HOP
    t = np.arange(n) / frontend.RATE
    x = 300 * rng.standard_normal(n) * (1 + np.sin(2 * np.pi * 4 * t))
    x[: frontend.HOP] = rng.integers(-3, 4, frontend.HOP)  # near-silent first frame
    k = 2 * frontend.HOP
    x[k : 2 * k] = 32767 * np.sin(2 * np.pi * 3000 * t[k : 2 * k])
    x[2 * k : 3 * k] = 32767 * np.sign(np.sin(2 * np.pi * 150 * t[2 * k : 3 * k]))
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


async def collect(dut, n_frames, out):
    for _ in range(n_frames):
        frame = np.zeros(frontend.N_BANDS, np.uint8)
        await RisingEdge(dut.feat_valid)
        for b in range(frontend.N_BANDS):
            await FallingEdge(dut.clk)
            assert int(dut.feat_valid.value) == 1 and int(dut.feat_band.value) == b
            frame[b] = int(dut.feat.value)
        out.append(frame)


@cocotb.test()
async def test_filterbank_bit_exact(dut):
    n_frames = 8
    pcm = stimulus(n_frames)
    want = frontend.filterbank(pcm)
    assert want.max() == frontend.FEAT_MAX, "stimulus should reach saturation"

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
        assert int(dut.busy.value) == 0, "previous sample not finished"
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
