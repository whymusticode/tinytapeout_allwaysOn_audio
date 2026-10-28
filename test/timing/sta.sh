#!/usr/bin/env bash
# Pre-layout timing of a block on gf180mcuD 7-track cells at the three corners the TT flow
# signs off (tt 25C 3.3V, ss 125C 3.0V, ff -40C 3.6V), for quick what-ifs before a CI run.
#   test/timing/sta.sh <top> "<chparams>" <verilog files...>
#   e.g. test/timing/sta.sh mac_harness "-set PIPE 0" src/mac.v test/timing/mac_harness.v
# Synthesis mirrors LibreLane's (yosys, ABC mapped to the tt library, area strategy). Wires are
# not modelled, so post-layout paths are slower: on this design at ss, reg-to-reg slack went
# from +17.9 ns pre-layout to -6.2 ns post-layout (ASIC_VERIFY.md, "Timing calibration").
# Needs: data/pdk (see ASIC_VERIFY.md), yosys, OpenROAD's sta (OPENSTA=path, or nix).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TOP=$1; CHP=$2; shift 2
LIBDIR=$ROOT/data/pdk/gf180_54435919/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/lib
L=$LIBDIR/gf180mcu_fd_sc_mcu7t5v0__
PERIOD=${PERIOD:-61}
OUT=${OUT:-$ROOT/data/timing/$TOP}
mkdir -p "$OUT"
STA=${OPENSTA:-$(nix build nixpkgs#openroad --no-link --print-out-paths 2>/dev/null)/bin/sta}
FILES=$(printf '%s ' "$@")
yosys -q -l "$OUT/yosys.log" -p "
  read_verilog -sv -I$ROOT/src $FILES
  ${CHP:+chparam $CHP $TOP}
  hierarchy -top $TOP
  synth -flatten -top $TOP
  check -assert
  dfflibmap -liberty ${L}tt_025C_3v30.lib
  abc -liberty ${L}tt_025C_3v30.lib
  opt_clean -purge
  hilomap -hicell gf180mcu_fd_sc_mcu7t5v0__tiel ZN -locell gf180mcu_fd_sc_mcu7t5v0__tiel ZN
  setundef -zero
  splitnets
  opt_clean -purge
  tee -o $OUT/stat.txt stat -liberty ${L}tt_025C_3v30.lib
  write_verilog -noattr $OUT/netlist.v"
for C in tt_025C_3v30 ss_125C_3v00 ff_n40C_3v60; do
  cat > "$OUT/sta_$C.tcl" <<TCL
read_liberty ${L}$C.lib
read_verilog $OUT/netlist.v
link_design $TOP
create_clock -name clk -period $PERIOD [get_ports clk]
set_clock_uncertainty 0.25 [get_clocks clk]
set_input_transition 0.5 [all_inputs]
set_load 0.02 [all_outputs]
# The synchronised reset drives every flop; unbuffered here (no resizer), it would swamp the
# report. Layout buffers it, and post_route_sta.sh times it for real.
foreach n {rst_n_s} { set d [get_nets -quiet \$n]; if {[llength \$d]} { set_false_path -from [get_cells -of_objects [get_pins -of_objects \$d -filter {direction == output}]] } }
report_checks -path_delay max -from [all_registers -clock_pins] -to [all_registers -data_pins] -format full_clock_expanded -digits 2 > $OUT/max_$C.rpt
report_checks -path_delay min -from [all_registers -clock_pins] -to [all_registers -data_pins] -digits 2 > $OUT/min_$C.rpt
exit
TCL
  "$STA" -no_splash -exit "$OUT/sta_$C.tcl" 2>&1 | grep -E "Error" || true
  echo "$C reg2reg slack: setup $(awk '/slack/{print $1; exit}' "$OUT/max_$C.rpt") ns, hold $(awk '/slack/{print $1; exit}' "$OUT/min_$C.rpt") ns"
done
grep -E "Chip area" "$OUT/stat.txt" | tail -1
