# Module-level cocotb tests, each checked bit-exact against model/frontend.py.
#   make            # all units
#   make filterbank # one unit
SIM ?= icarus
TOPLEVEL_LANG ?= verilog
SRC_DIR = $(PWD)/../../src
COMPILE_ARGS += -I$(SRC_DIR)
UNITS = filterbank audio_ring mac

all: $(UNITS)

# FB_BANDS=24 make filterbank   (also FB_ORDER=2)
FB_BANDS ?= 12
FB_ORDER ?= 4
export FB_BANDS FB_ORDER
filterbank:
	$(MAKE) -f $(lastword $(MAKEFILE_LIST)) sim SIM_BUILD=sim_build/$@_b$(FB_BANDS)_o$(FB_ORDER) TOPLEVEL=$@ \
	  COCOTB_TEST_MODULES=test_$@ VERILOG_SOURCES="$(SRC_DIR)/filterbank.v" \
	  UNIT_PARAMS="-Pfilterbank.N_BANDS=$(FB_BANDS) -Pfilterbank.ORDER=$(FB_ORDER)"

audio_ring:
	$(MAKE) -f $(lastword $(MAKEFILE_LIST)) sim SIM_BUILD=sim_build/$@ TOPLEVEL=$@ COCOTB_TEST_MODULES=test_$@ \
	  VERILOG_SOURCES="$(SRC_DIR)/audio_ring.v" \
	  UNIT_PARAMS="-Paudio_ring.BUF_LOG2=8 -Paudio_ring.PRE_LOG2=6 -Paudio_ring.INIT_CYCLES=100"

# MAC_PIPE=1 make mac   tests the pipelined variant
MAC_PIPE ?= 0
MAC_ACC_W ?= 24
export MAC_PIPE MAC_ACC_W
mac:
	$(MAKE) -f $(lastword $(MAKEFILE_LIST)) sim SIM_BUILD=sim_build/$@_p$(MAC_PIPE) TOPLEVEL=$@ COCOTB_TEST_MODULES=test_$@ \
	  VERILOG_SOURCES="$(SRC_DIR)/mac.v" UNIT_PARAMS="-Pmac.PIPE=$(MAC_PIPE) -Pmac.ACC_W=$(MAC_ACC_W)"

.PHONY: all $(UNITS)

ifneq ($(TOPLEVEL),)
COMPILE_ARGS += $(UNIT_PARAMS)
include $(shell cocotb-config --makefiles)/Makefile.sim
endif
