![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# Open always-on wake-word detector (Tiny Tapeout, sky130)

The goal is an open-source alternative to the always-on "hey Google" style hotword chips, for Linux phones, smart speakers,
handhelds and embedded Linux. The chip listens to a PDM microphone at very low power and raises an interrupt
when it hears a keyword.

Datasheet: [docs/info.md](docs/info.md)

## Architecture (target)

```
PDM mic -> CIC /64 -> 16 kHz PCM -> mel filterbank (time-multiplexed biquads, bit-serial MAC)
        -> log energies per 16 ms frame -> streaming sparse ternary classifier (DeltaGRU / TCN)
        -> score + threshold + hysteresis -> WAKE_IRQ
                         \-> energy VAD gates the classifier
weights: (A) hard-wired sparse ternary ROM, or (B) streamed from the QSPI flash/PSRAM Pmod
```

Audio is slow and the clock is fast: there are more than 1,000 clock cycles per audio sample. So one
shared bit-serial MAC can do all the arithmetic, and on-chip storage, not compute, sets the limit.

| Block | Status |
|---|---|
| PDM clock + 3rd-order CIC (`src/pdm_cic.v`) | done, bit-exact vs `model/frontend.py` |
| Energy VAD (`src/vad.v`) | done, bit-exact |
| Mel filterbank + log energy | next |
| Model feasibility study (param budget vs. accuracy) | next |
| Streaming classifier, SPI host interface, QSPI weights | planned |

## Layout

- `src/`: Verilog. The top module is `tt_um_whymusticode_wakeword` in `src/project.v`.
- `model/`: bit-exact Python reference models. Training code will also go here.
- `test/`: cocotb tests that compare the RTL against `model/`.
- `docs/info.md`: the Tiny Tapeout datasheet.

## Running the tests

```sh
. ./env.sh          # creates .venv (cocotb, numpy, scipy); on NixOS also sets LD_LIBRARY_PATH
cd test && make     # RTL simulation with Icarus Verilog
```

Gate-level tests run automatically in the GitHub `gds` action.

## Pin map

| Pin | Function |
|---|---|
| `ui[0]` | PDM_DATA |
| `ui[1]` | PDM_EDGE: 0 = data valid while PDM_CLK high, 1 = while low |
| `ui[4:2]` | host SPI SCK / CS_N / MOSI (reserved) |
| `ui[7:5]` | VAD threshold for bring-up (× 512) |
| `uo[0]` | WAKE_IRQ (reserved) |
| `uo[1]` | PDM_CLK (clk / 16) |
| `uo[2]` | host SPI MISO (reserved) |
| `uo[3]` | VAD |
| `uio[7:0]` | QSPI Pmod: weight flash / PSRAM (reserved, Tiny Tapeout qspi-pmod pinout) |

## What is Tiny Tapeout?

Tiny Tapeout is an educational project that makes it easier and cheaper than ever to get digital and analog
designs manufactured on a real chip. See https://tinytapeout.com.
