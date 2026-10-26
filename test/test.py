# SPDX-FileCopyrightText: © 2026 Michael Benton
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
import frontend  # noqa: E402

CLK_PERIOD_NS = 61  # ~16.384 MHz
GL = os.environ.get("GATES") == "yes"


async def reset(dut, ui=0):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())
    dut.ena.value = 1
    dut.ui_in.value = ui
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1


def pdm_clk(dut):
    return (int(dut.uo_out.value) >> 1) & 1


async def drive_pdm(dut, bits, edge, ui_hi):
    """Act as the mic: put each bit on PDM_DATA at the start of its valid half-cycle."""
    want = 0 if edge else 1
    prev = pdm_clk(dut)
    while True:  # find the edge that starts the valid half, then step one PDM period per bit
        await RisingEdge(dut.clk)
        cur = pdm_clk(dut)
        if cur == want and prev != want:
            break
        prev = cur
    for b in bits:
        dut.ui_in.value = ui_hi | (edge << 1) | int(b)
        await ClockCycles(dut.clk, 16)


def tone_bits(freq, amp, n_pcm, seed=0):
    n = n_pcm * frontend.DECIM
    t = np.arange(n) / frontend.PDM_RATE
    return frontend.pdm_modulate(amp * np.sin(2 * np.pi * freq * t), seed)


async def capture_pcm(dut, n):
    """Record PCM samples from the CIC (RTL only; internal signals)."""
    out = []
    cic = dut.user_project.u_cic
    while len(out) < n:
        await RisingEdge(dut.clk)
        if int(cic.pcm_valid.value):
            out.append(cic.pcm.value.to_signed())
    return np.array(out, dtype=np.int16)


@cocotb.test()
async def test_pdm_clock(dut):
    await reset(dut)
    await ClockCycles(dut.clk, 3)
    seen = [pdm_clk(dut)]
    for _ in range(63):
        await RisingEdge(dut.clk)
        seen.append(pdm_clk(dut))
    # clk/16: 8 high, 8 low
    edges = sum(1 for a, b in zip(seen, seen[1:]) if a != b)
    assert edges == 8, f"expected 8 PDM clock edges in 64 cycles, saw {edges}"


@cocotb.test(skip=GL)
@cocotb.parametrize(edge=[0, 1])
async def test_cic_bit_exact(dut, edge):
    await reset(dut, ui=edge << 1)
    n_pcm = 48
    bits = tone_bits(1000, 0.5, n_pcm, seed=edge)
    cocotb.start_soon(drive_pdm(dut, bits, edge, 0))
    got = await capture_pcm(dut, n_pcm)
    # With edge=1 the first valid half-cycle starts one PDM period after reset, so the
    # CIC has already taken one sample of the idle (0) data pin.
    want = frontend.cic3(np.concatenate([[0] * edge, bits]))[:n_pcm]
    dut._log.info(f"edge={edge} rtl={got[:8]} ref={want[:8]}")
    assert np.array_equal(got, want), f"edge={edge}: first mismatch at {np.argmax(got != want)}"
    # Past the CIC warm-up, the tone should be at roughly the input level.
    assert 0.35 * 32768 < np.abs(got[8:]).max() < 0.65 * 32768


@cocotb.test()
async def test_vad(dut):
    thr_code = 2  # threshold = 1024
    ui_hi = thr_code << 5
    await reset(dut, ui=ui_hi)
    # 20 ms silence, 20 ms of tone at -10 dBFS, 30 ms silence.
    quiet = np.zeros(320 * frontend.DECIM)
    t = np.arange(320 * frontend.DECIM) / frontend.PDM_RATE
    loud = 0.3 * np.sin(2 * np.pi * 700 * t)
    x = np.concatenate([quiet, loud, np.zeros(480 * frontend.DECIM)])
    bits = frontend.pdm_modulate(x, seed=3)
    want_vad, _ = frontend.vad(frontend.cic3(bits), thr_code * 512)

    cocotb.start_soon(drive_pdm(dut, bits, 0, ui_hi))
    got = []
    if GL:
        # No internal signals: poll the pin once per PCM period.
        for _ in range(len(want_vad) - 1):
            await ClockCycles(dut.clk, frontend.DECIM * 16)
            got.append((int(dut.uo_out.value) >> 3) & 1)
        got = np.array(got)
        assert got[:300].sum() == 0 and got[360:620].all() and got[-60:].sum() == 0
        return
    # RTL: VAD updates on the clock after each PCM sample; compare bit-exact.
    cic = dut.user_project.u_cic
    while len(got) < len(want_vad):
        await RisingEdge(dut.clk)
        if int(cic.pcm_valid.value):
            await RisingEdge(dut.clk)
            await ReadOnly()
            got.append((int(dut.uo_out.value) >> 3) & 1)
    got = np.array(got)
    dut._log.info(f"VAD on for {got.sum()} of {len(got)} samples (ref {want_vad.sum()})")
    assert want_vad[:320].sum() == 0 and want_vad[340:640].all() and want_vad[-100:].sum() == 0
    assert np.array_equal(got, want_vad), f"VAD mismatches at {np.nonzero(got != want_vad)[0]}"


@cocotb.test()
async def test_psram_buffer_init(dut):
    """The rolling audio buffer resets PSRAM A (0x66, 0x99) after power-up, then starts
    writing samples; the flash and PSRAM B stay deselected. (Streaming itself is checked
    bit-exact in test/unit/test_audio_ring.py with a PSRAM model.)"""
    await reset(dut)
    cmds, bits, prev_sck, n_cycles = [], [], 0, 0
    while len(cmds) < 3 and n_cycles < 8000:
        await RisingEdge(dut.clk)
        n_cycles += 1
        uo = int(dut.uio_out.value)
        assert int(dut.uio_oe.value) == 0b1111_1011
        assert uo & 1 and uo >> 7 & 1, "flash / PSRAM B must stay deselected"
        cs, sck, mosi = uo >> 6 & 1, uo >> 3 & 1, uo >> 1 & 1
        if not cs and sck and not prev_sck:
            bits.append(mosi)
        if cs and bits:
            cmds.append(int("".join(map(str, bits[:8])), 2))
            bits = []
        prev_sck = sck
    dut._log.info(f"PSRAM commands after reset: {[hex(c) for c in cmds]} ({n_cycles} cycles)")
    assert cmds[:2] == [0x66, 0x99], cmds
    assert cmds[2] == 0x02, "expected the first sample write after the reset sequence"
