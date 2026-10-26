#!/usr/bin/env bash
# Run a command in a memory-capped systemd scope so a runaway job is OOM-killed
# instead of freezing the machine. Prints the peak memory of the whole process tree.
#   MEM=60G model/capped.sh python model/build_cache.py
set -euo pipefail
MEM=${MEM:-60G}
unit=capped-$$-$RANDOM
peakf=$(mktemp)
echo 0 > "$peakf"
(  # sample the scope's total memory once a second
  peak=0
  while sleep 1; do
    cur=$(systemctl --user show "$unit.scope" -p MemoryCurrent --value 2>/dev/null) || continue
    [[ $cur =~ ^[0-9]+$ ]] && (( cur > peak )) && peak=$cur && echo $peak > "$peakf"
  done
) &
sampler=$!
status=0
systemd-run --user --scope --quiet --unit="$unit" -p MemoryMax="$MEM" -p MemorySwapMax=0 \
  -E LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" -- "$@" || status=$?
kill $sampler 2>/dev/null || true
echo "[capped.sh] exit $status, peak memory $(( $(cat "$peakf") / 1024 / 1024 )) MiB (cap $MEM)" >&2
rm -f "$peakf"
exit $status
