#!/usr/bin/env bash
# Plan A (prereg-v1.1): sizes xs and s on the Mac (mps), 2 jobs, low priority. Runs the sweep
# only on mains power: kills it on battery, restarts it on AC (cells resume from their last
# checkpoint). Exits when both phases report nothing left to run.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
unset RT_CONFIG RT_EVAL_DIR
LOG=results/mac_sweep.log
on_ac() { pmset -g batt | grep -q "AC Power"; }
remaining() { uv run python -m rt.sweep --device mps --only "$1" --dry-run 2>/dev/null | grep -c '^A-'; }
for glob in 'A-xs-*' 'A-s-*'; do
  while [ "$(remaining "$glob")" -gt 0 ]; do
    if ! on_ac; then echo "$(date -u +%FT%TZ) on battery, waiting" >> $LOG; sleep 120; continue; fi
    echo "$(date -u +%FT%TZ) start $glob" >> $LOG
    nice -n 15 uv run python -m rt.sweep --device mps --only "$glob" --jobs 2 >> $LOG 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      if ! on_ac; then
        echo "$(date -u +%FT%TZ) battery: stopping sweep" >> $LOG
        pkill -P $pid; kill $pid; pkill -f "rt.train --cell A-(xs|s)-"; pkill -f "rt.evaluate --cell A-(xs|s)-"
        break
      fi
      sleep 30
    done
    wait $pid 2>/dev/null
  done
done
echo "$(date -u +%FT%TZ) MAC SWEEP DONE" >> $LOG
