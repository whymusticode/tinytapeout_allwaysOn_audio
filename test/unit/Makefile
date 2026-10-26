# Module-level cocotb tests, each checked bit-exact against model/frontend.py.
#   make            # all units
#   make filterbank # one unit
SIM ?= icarus
TOPLEVEL_LANG ?= verilog
SRC_DIR = $(PWD)/../../src
COMPILE_ARGS += -I$(SRC_DIR)
UNITS = filterbank

all: $(UNITS)

filterbank:
	$(MAKE) -f $(lastword $(MAKEFILE_LIST)) sim SIM_BUILD=sim_build/$@ TOPLEVEL=$@ COCOTB_TEST_MODULES=test_$@ \
	  VERILOG_SOURCES="$(SRC_DIR)/filterbank.v"

.PHONY: all $(UNITS)

ifneq ($(TOPLEVEL),)
include $(shell cocotb-config --makefiles)/Makefile.sim
endif
