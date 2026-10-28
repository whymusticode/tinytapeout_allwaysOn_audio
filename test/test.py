# SPDX-FileCopyrightText: © 2026 Michael Benton
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
import frontend  # noqa: E402
import frontend_var as FV  # noqa: E402

CLK_PERIOD_NS = 61  # ~16.384 MHz
GL = os.environ.get("GATES") == "yes"
N_BANDS = 12  # project.v
EDGE_BIT, EXT_BIT = 7, 6  # ui straps
SPI_HALF = 8  # host SCK half period in clocks (clk / 16)


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
        dut.ui_in.value = ui_hi | (edge << EDGE_BIT) | int(b)
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
    await ClockCycles(dut.clk, 6)  # rst_n is synchronised: the design leaves reset 2 clocks later
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
    await reset(dut, ui=edge << EDGE_BIT)
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
    thr_code = 2  # threshold = 1024, the VAD_THR register's reset value
    ui_hi = 0
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


# ---- host SPI (spi_cfg.v): bit-banged mode 0 on ui[3] CS_N, ui[4] SCK, ui[5] MOSI, uo[5] MISO ----

CS, SCK, MOSI = 3, 4, 5


_ui = {}  # shadow of ui_in: several bits change in one time step, and a read-back there is stale


def _set(dut, bit, v):
    u = _ui.get("v", 0)
    _ui["v"] = u = (u | (1 << bit)) if v else (u & ~(1 << bit))
    dut.ui_in.value = u


async def spi(dut, out_bytes):
    """One transaction; returns the bytes read on MISO."""
    got = []
    _set(dut, CS, 0)
    await ClockCycles(dut.clk, SPI_HALF)
    for byte in out_bytes:
        r = 0
        for i in range(8):
            _set(dut, MOSI, byte >> (7 - i) & 1)
            await ClockCycles(dut.clk, SPI_HALF)
            _set(dut, SCK, 1)
            r = r << 1 | (int(dut.uo_out.value) >> 5 & 1)
            await ClockCycles(dut.clk, SPI_HALF)
            _set(dut, SCK, 0)
        got.append(r)
    await ClockCycles(dut.clk, SPI_HALF)
    _set(dut, CS, 1)
    await ClockCycles(dut.clk, SPI_HALF)
    return got


async def reg_write(dut, addr, data):
    await spi(dut, [addr & 0x7F] + list(data))


async def reg_read(dut, addr, n=1):
    return (await spi(dut, [0x80 | addr] + [0] * n))[1:]


async def reset_idle(dut, ui=0):
    _ui["v"] = ui | (1 << CS)  # CS_N idle high
    await reset(dut, ui=_ui["v"])
    await ClockCycles(dut.clk, 5)


@cocotb.test()
async def test_spi_registers(dut):
    await reset_idle(dut)
    assert await reg_read(dut, 0x00) == [0x57], "ID"
    assert await reg_read(dut, 0x01, 6) == [0x01, 0x00, 0x04, 64, 3, 0], "reset values"
    await reg_write(dut, 0x02, [0x34, 0x12, 128, 6])
    assert await reg_read(dut, 0x02, 4) == [0x34, 0x12, 128, 6]
    st = (await reg_read(dut, 0x10))[0]
    assert st >> 2 & 3 == 0, f"straps read back wrong: {st:#x}"


@cocotb.test()
async def test_inject_features(dut):
    """Host-injected PCM (mic and CIC bypassed) -> filterbank -> feature frame over SPI,
    bit-exact with frontend_var at the chip's widths."""
    await reset_idle(dut)
    await reg_write(dut, 0x01, [0x03])  # miso_late, inject
    rng = np.random.default_rng(5)
    n = 2 * 256
    t = np.arange(n) / 16000
    pcm = np.clip(np.round(8000 * np.sin(2 * np.pi * 900 * t) + 500 * rng.standard_normal(n)), -32768, 32767).astype(np.int16)
    for x in pcm:
        u = int(x) & 0xFFFF
        await reg_write(dut, 0x08, [u & 0xFF, u >> 8])
        await ClockCycles(dut.clk, 1024 - 3 * 8 * 2 * SPI_HALF - 4 * SPI_HALF + 16)
    await ClockCycles(dut.clk, 1100)
    st, = await reg_read(dut, 0x10)
    assert not st >> 4 & 1, "inject overrun"
    cnt = await reg_read(dut, 0x15, 2)
    assert cnt == [n % 256, 2], f"PCM / frame counters {cnt}"
    got = np.array(await reg_read(dut, 0x20, N_BANDS))
    v = FV.variant(**FV.LEAN5, bands=N_BANDS, order=4, hop=256)
    x = ((pcm.astype(np.int64) >> 4) << 4)[None]
    g, a1, a2 = FV.coeffs_int(v)
    out = np.zeros((1, 2, N_BANDS), np.int64)
    FV._fb_batch(x, g, a1, a2, 256, v["coef_frac"], v["y_w"], v["guard"], v["acc_w"], v["acc_shift"],
                 v["log_frac"], True, out)
    lo, levels = FV.code_range(v)
    want = np.clip(out[0, 1] - lo, 0, levels - 1)
    dut._log.info(f"frame 1: rtl {got.tolist()} ref {want.tolist()}")
    assert np.array_equal(got, want)


@cocotb.test(skip=GL)
async def test_external_mic_clock(dut):
    """PDM_EXT = 1: the mic clock comes from elsewhere (a phone's codec) at 2.048 MHz, not a
    multiple of our clock, and the host sets R = 128, shift 6 for 16 kHz."""
    await reset_idle(dut, ui=1 << EXT_BIT)
    await reg_write(dut, 0x04, [128, 6])
    n_pcm = 24
    rate = 2_048_000
    t = np.arange(n_pcm * 128) / rate
    bits = frontend.pdm_modulate(0.5 * np.sin(2 * np.pi * 1000 * t), seed=7)
    half_ps = round(1e12 / rate / 2)

    async def mic():
        # data changes just after the rising edge, valid while the clock is high (PDM_EDGE 0)
        for b in bits:
            _set(dut, 1, 1)
            await Timer(15_000, unit="ps")
            _set(dut, 0, int(b))
            await Timer(half_ps - 15_000, unit="ps")
            _set(dut, 1, 0)
            await Timer(half_ps, unit="ps")

    cocotb.start_soon(mic())
    got = await capture_pcm(dut, n_pcm)
    want = frontend.cic3(bits, decim=128, shift=6)[:n_pcm]
    dut._log.info(f"rtl {got[:8]} ref {want[:8]}")
    assert np.array_equal(got, want), f"first mismatch at {np.argmax(got != want)}"
    assert 0.35 * 32768 < np.abs(got[8:]).max() < 0.65 * 32768
