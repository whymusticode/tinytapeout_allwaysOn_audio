"""audio_ring.v: PCM ring buffer in SPI PSRAM, streamed to the host over UART on request.

Small parameters for simulation speed (see the Makefile): 256-sample ring, 64-sample
pre-roll. The PSRAM is a behavioural APS6404L (SPI mode, commands 0x66/0x99/0x02/0x03)
that also checks the device rules the RTL must respect: reset before use, CE# low at most
8 us, mode-0 timing. The UART receiver decodes 8N1 at clk/16.

Checks: the stream starts with "WAKE", then the samples from PRE before the request,
contiguous, continuing with live samples until the request drops; a second request
starts a fresh session.
"""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

CLK_HZ = 16_384_000
PRE = 64
BAUD_CLKS = 16
SAMPLE_EVERY = 1024  # clock cycles between PCM samples, as in the chip (16 kHz)
CE_MAX_CLKS = int(8e-6 * CLK_HZ)  # APS6404L tCEM


class PSRAM:
    def __init__(self, dut):
        self.dut = dut
        self.mem = {}
        self.reset_seen = False
        self.max_ce = 0
        self.errors = []

    async def run(self):
        dut = self.dut
        while True:
            await FallingEdge(dut.ram_cs_n)
            t0 = cocotb.utils.get_sim_time("ns")
            bits_in, out_bits = [], None
            while True:
                # wait for a rising SCK edge or CS going high
                await RisingEdge(dut.clk)
                if int(dut.ram_cs_n.value):
                    break
                if int(dut.spi_sck.value) and not getattr(self, "_sck", 0):
                    # SCK rose on the previous clock; it falls on this one. Sample MOSI, and
                    # (reads) drive the next data bit now, as the device does on the falling edge.
                    bits_in.append(int(dut.spi_mosi.value))
                    if out_bits is None and len(bits_in) == 32 and self._byte(bits_in[:8]) == 0x03:
                        addr = self._val(bits_in[8:32])
                        word = self.mem.get(addr, 0) | (self.mem.get(addr + 1, 0) << 8)
                        out_bits = [(word >> (15 - i)) & 1 for i in range(16)]
                    if out_bits is not None:
                        k = len(bits_in) - 32
                        dut.spi_miso.value = out_bits[k] if 0 <= k < 16 else 0
                self._sck = int(dut.spi_sck.value)
            self._sck = 0
            ce = (cocotb.utils.get_sim_time("ns") - t0) * 1e-9 * CLK_HZ
            self.max_ce = max(self.max_ce, ce)
            self._finish(bits_in)

    @staticmethod
    def _val(bits):
        v = 0
        for b in bits:
            v = (v << 1) | b
        return v

    def _byte(self, bits):
        return self._val(bits)

    def _finish(self, bits):
        if len(bits) < 8:
            self.errors.append(f"short transaction ({len(bits)} bits)")
            return
        cmd = self._byte(bits[:8])
        if cmd in (0x66, 0x99):
            self.reset_seen = self.reset_seen or cmd == 0x99
            return
        if not self.reset_seen:
            self.errors.append(f"command {cmd:#x} before reset")
        if cmd == 0x02:
            addr = self._val(bits[8:32])
            data = self._val(bits[32:48])
            # the RTL sends the sample big-endian on the wire; store little-endian bytes
            self.mem[addr] = data & 0xFF
            self.mem[addr + 1] = data >> 8
            if len(bits) != 48:
                self.errors.append(f"write with {len(bits)} bits")
        elif cmd != 0x03:
            self.errors.append(f"unexpected command {cmd:#x}")


async def uart_rx(dut, out):
    """Collect bytes from dut.uart_tx (8N1, BAUD_CLKS per bit)."""
    while True:
        await FallingEdge(dut.uart_tx)
        await ClockCycles(dut.clk, BAUD_CLKS // 2)
        assert int(dut.uart_tx.value) == 0, "bad start bit"
        b = 0
        for i in range(8):
            await ClockCycles(dut.clk, BAUD_CLKS)
            b |= int(dut.uart_tx.value) << i
        await ClockCycles(dut.clk, BAUD_CLKS)
        assert int(dut.uart_tx.value) == 1, "bad stop bit"
        out.append(b)


async def feed(dut, samples, written):
    for s in samples:
        dut.pcm.value = s & 0xFFFF
        dut.pcm_valid.value = 1
        await RisingEdge(dut.clk)
        dut.pcm_valid.value = 0
        written.append(s)
        await ClockCycles(dut.clk, SAMPLE_EVERY - 1)


def decode(stream):
    assert bytes(stream[:4]) == b"WAKE", f"header {bytes(stream[:4])!r}"
    body = stream[4:]
    vals = [body[i] | (body[i + 1] << 8) for i in range(0, len(body) - 1, 2)]
    return [v - 65536 if v >= 32768 else v for v in vals]


def check_session(got, written, n_at_req):
    """got must be a contiguous run of `written` starting PRE samples before the request."""
    start = n_at_req - PRE
    assert len(got) > PRE + 20, f"only {len(got)} samples streamed"
    # the request is synchronised and may land a sample later than our count
    for s0 in (start, start + 1):
        if got == written[s0 : s0 + len(got)]:
            return s0
    first_bad = next(i for i, (a, b) in enumerate(zip(got, written[start:])) if a != b)
    raise AssertionError(f"stream diverges at sample {first_bad}: got {got[first_bad:first_bad+4]}, "
                         f"expected {written[start + first_bad:start + first_bad + 4]}")


@cocotb.test()
async def test_ring_stream(dut):
    cocotb.start_soon(Clock(dut.clk, 61, unit="ns").start())
    dut.rst_n.value = 0
    dut.pcm.value = 0
    dut.pcm_valid.value = 0
    dut.stream_req.value = 0
    dut.spi_miso.value = 0
    ram = PSRAM(dut)
    cocotb.start_soon(ram.run())
    rx = []
    cocotb.start_soon(uart_rx(dut, rx))
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1

    rng = random.Random(1)
    samples = [rng.randint(-32768, 32767) for _ in range(900)]
    written = []
    cocotb.start_soon(feed(dut, samples, written))

    for session in range(2):
        await ClockCycles(dut.clk, SAMPLE_EVERY * (200 if session == 0 else 120))
        n_at_req = len(written)
        rx.clear()
        dut.stream_req.value = 1
        await ClockCycles(dut.clk, SAMPLE_EVERY * 150)  # streams pre-roll, then catches up with live audio
        dut.stream_req.value = 0
        await ClockCycles(dut.clk, BAUD_CLKS * 40)
        # write path: the PSRAM must hold the most recent samples at ring index i % 256
        ring = 256
        stored = []
        for i in range(max(0, len(written) - ring + 2), len(written) - 1):
            a = 2 * (i % ring)
            v = ram.mem.get(a, 0) | (ram.mem.get(a + 1, 0) << 8)
            stored.append(v - 65536 if v >= 32768 else v)
        exp = written[max(0, len(written) - ring + 2) : len(written) - 1]
        bad = [i for i, (x, y) in enumerate(zip(stored, exp)) if x != y]
        assert not bad, f"PSRAM contents wrong at {len(bad)} places, first: stored {stored[bad[0]]} expected {exp[bad[0]]}"
        got = decode(rx)
        s0 = check_session(got, written, n_at_req)
        lag = len(written) - (s0 + len(got))
        dut._log.info(f"session {session}: {len(got)} samples from index {s0} (request at {n_at_req}); "
                      f"ended {lag} samples behind live")
        assert lag <= 3, f"stream did not catch up with live audio ({lag} samples behind)"
        assert int(dut.streaming.value) == 0

    assert not ram.errors, ram.errors
    assert ram.max_ce <= CE_MAX_CLKS, f"CE# low {ram.max_ce:.0f} clocks > {CE_MAX_CLKS} (8 us)"
    dut._log.info(f"max CE# low time {ram.max_ce:.0f} clocks ({ram.max_ce / CLK_HZ * 1e6:.2f} us)")
