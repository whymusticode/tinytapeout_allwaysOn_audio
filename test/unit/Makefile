# Module-level cocotb tests, each checked bit-exact against model/frontend.py.
#   make            # all units
#   make filterbank # one unit
SIM ?= icarus
TOPLEVEL_LANG ?= verilog
SRC_DIR = $(PWD)/../../src
COMPILE_ARGS += -I$(SRC_DIR)
UNITS = filterbank audio_ring

all: $(UNITS)

filterbank:
	$(MAKE) -f $(lastword $(MAKEFILE_LIST)) sim SIM_BUILD=sim_build/$@ TOPLEVEL=$@ COCOTB_TEST_MODULES=test_$@ \
	  VERILOG_SOURCES="$(SRC_DIR)/filterbank.v"

audio_ring:
	$(MAKE) -f $(lastword $(MAKEFILE_LIST)) sim SIM_BUILD=sim_build/$@ TOPLEVEL=$@ COCOTB_TEST_MODULES=test_$@ \
	  VERILOG_SOURCES="$(SRC_DIR)/audio_ring.v" \
	  UNIT_PARAMS="-Paudio_ring.BUF_LOG2=8 -Paudio_ring.PRE_LOG2=6 -Paudio_ring.INIT_CYCLES=100"

.PHONY: all $(UNITS)

ifneq ($(TOPLEVEL),)
COMPILE_ARGS += $(UNIT_PARAMS)
include $(shell cocotb-config --makefiles)/Makefile.sim
endif
