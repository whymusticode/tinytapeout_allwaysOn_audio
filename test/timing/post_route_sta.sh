#!/usr/bin/env bash
# Post-route timing from a GDS CI run's artifacts (netlist + extracted parasitics), with the
# worst setup/hold slack per instance prefix, at the three sign-off corners.
#   test/timing/post_route_sta.sh <gh run id> [instance prefix ...]
#   e.g. test/timing/post_route_sta.sh 40721728925 u_cic. pcm u_ring. u_vad.
# A prefix matches the output net names of flops (u_cic.i3[1]); flops that drive a module port
# take the top-level net name instead (pcm[11] is u_cic.pcm).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=$1; shift
OUT=$ROOT/data/timing/ci_$RUN
L=$ROOT/data/pdk/gf180_54435919/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/lib/gf180mcu_fd_sc_mcu7t5v0__
STA=${OPENSTA:-$(nix build nixpkgs#openroad --no-link --print-out-paths 2>/dev/null)/bin/sta}
[ -d "$OUT/GDS_logs" ] || gh run download "$RUN" -R whymusticode/tinytapeout_allwaysOn_audio -n GDS_logs -D "$OUT/GDS_logs" >/dev/null
F=$OUT/GDS_logs/runs/wokwi/final
TOP=tt_um_whymusticode_wakeword
for C in tt_025C_3v30 ss_125C_3v00 ff_n40C_3v60; do
  case $C in tt*) S=nom;; ss*) S=max;; ff*) S=min;; esac  # the corner that binds each check
  SPEF=$(ls "$F"/spef/$S/*.spef | head -1)
  {
    echo "read_liberty ${L}$C.lib"
    echo "read_verilog $F/nl/$TOP.nl.v"
    echo "link_design $TOP"
    echo "read_spef $SPEF"
    echo "read_sdc $(ls "$F"/sdc/*.sdc | head -1)"
    echo "set_propagated_clock [all_clocks]"
    echo "proc ws {kind pins} { if {[llength \$pins] == 0} { return n/a }; set p [find_timing_paths -path_delay \$kind -to \$pins -group_path_count 1]; if {[llength \$p] == 0} { return n/a }; return [format %.2f [get_property [lindex \$p 0] slack]] }"
    # flops keep their RTL names only on their output nets (e.g. u_cic.i3[1])
    echo "proc regs_of {prefix} { set d {}; foreach c [all_registers -cells] { foreach q [get_pins -of_objects \$c -filter {direction == output}] { set n [get_nets -quiet -of_objects \$q]; if {\$n ne {} && [string match \"\${prefix}*\" [get_full_name \$n]]} { lappend d [get_pins -of_objects \$c -filter {name == D}] } } }; return \$d }"
    echo "puts \"$C all reg2reg: setup [ws max [all_registers -data_pins]] hold [ws min [all_registers -data_pins]]\""
    for I in "$@"; do
      echo "set d [regs_of $I]"
      echo "puts \"$C $I: setup [ws max \$d] hold [ws min \$d]\""
    done
    echo "report_checks -path_delay max -format full_clock_expanded -digits 2 > $OUT/max_$C.rpt"
  } > "$OUT/sta_$C.tcl"
  "$STA" -no_splash -exit "$OUT/sta_$C.tcl" 2>&1 | grep -vE "^(Warning|$)" | grep -E "setup|Error" || true
done
