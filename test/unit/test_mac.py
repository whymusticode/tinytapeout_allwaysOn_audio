"""mac.v: signed 8x8 multiply into a wrapping accumulator, one product per clock.

Random a, w, en, clr every clock (including the corner operands -128 and 127), compared each
clock against a Python model with the same wrap-around at ACC_W bits. With PIPE = 1 the model's
inputs are delayed one clock. Also checks a full-length dot product (511 terms of
-128 * -128 = +2^14, the most that fit ACC_W = 24) and the reset value.
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, ReadOnly

ACC_W = int(os.environ.get("MAC_ACC_W", "24"))
PIPE = int(os.environ.get("MAC_PIPE", "0"))
CORNERS = [-128, -127, -1, 0, 1, 127]


def wrap(x):
    x &= (1 << ACC_W) - 1
    return x - (1 << ACC_W) if x >> (ACC_W - 1) else x


class Model:
    def __init__(self):
        self.acc = 0
        self.q = [(0, 0, 0, 0)] * PIPE  # (en, clr, a, w) in the pipeline

    def step(self, en, clr, a, w):
        self.q.append((en, clr, a, w))
        en, clr, a, w = self.q.pop(0)
        if en:
            self.acc = wrap((0 if clr else self.acc) + a * w)
        elif clr:
            self.acc = 0
        return self.acc


async def start(dut):
    cocotb.start_soon(Clock(dut.clk, 61, unit="ns").start())
    dut.rst_n.value = 0
    dut.en.value = 0
    dut.clr.value = 0
    dut.a.value = 0
    dut.w.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert dut.acc.value.to_signed() == 0


@cocotb.test()
async def random_ops(dut):
    await start(dut)
    m = Model()
    rng = random.Random(1)
    # Drive on the falling edge so each rising edge samples one set of inputs.
    for i in range(20000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        got = dut.acc.value.to_signed()
        assert got == m.acc, f"step {i}: acc {got} != model {m.acc}"
        await cocotb.triggers.FallingEdge(dut.clk)
        pick = lambda: rng.choice(CORNERS) if rng.random() < 0.2 else rng.randint(-128, 127)
        en, clr = int(rng.random() < 0.9), int(rng.random() < 0.05)
        a, w = pick(), pick()
        dut.en.value, dut.clr.value, dut.a.value, dut.w.value = en, clr, a & 0xFF, w & 0xFF
        m.step(en, clr, a, w)


@cocotb.test()
async def longest_dot_product(dut):
    await start(dut)
    n = (1 << (ACC_W - 15)) - 1  # most terms of +2^14 that stay below 2^(ACC_W-1)
    await cocotb.triggers.FallingEdge(dut.clk)
    for i in range(n):
        dut.en.value, dut.clr.value, dut.a.value, dut.w.value = 1, int(i == 0), 0x80, 0x80
        await cocotb.triggers.FallingEdge(dut.clk)
    dut.en.value = 0
    dut.clr.value = 0
    await ClockCycles(dut.clk, PIPE + 1)
    await ReadOnly()
    want = n * 16384
    assert dut.acc.value.to_signed() == want, f"{dut.acc.value.to_signed()} != {want}"
