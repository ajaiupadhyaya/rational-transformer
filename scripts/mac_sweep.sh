#!/usr/bin/env bash
# Plan A (prereg-v1.1): sizes xs and s on the Mac (mps), 2 jobs, low priority. Runs the sweep
# only on mains power: kills it on battery, restarts it on AC (cells resume from their last
# checkpoint). Exits when both phases report nothing left to run.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
unset RT_CONFIG RT_EVAL_DIR
LOG=results/mac_sweep.log
# ALLOW_BATTERY_FLOOR=N (owner's one-off permission): also run on battery while charge > N%.
on_ac() {
  pmset -g batt | grep -q "AC Power" && return 0
  [ -n "${ALLOW_BATTERY_FLOOR:-}" ] || return 1
  pct=$(pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d %)
  [ "${pct:-0}" -gt "$ALLOW_BATTERY_FLOOR" ]
}
remaining() { uv run python -m rt.sweep --device mps --only "$1" --dry-run 2>/dev/null | grep -c '^A-'; }
for glob in 'A-xs-*' 'A-s-*'; do
  while [ "$(remaining "$glob")" -gt 0 ]; do
    if ! on_ac; then echo "$(date -u +%FT%TZ) on battery, waiting" >> $LOG; sleep 120; continue; fi
    echo "$(date -u +%FT%TZ) start $glob" >> $LOG
    nice -n 15 uv run python -m rt.sweep --device mps --only "$glob" --jobs 2 --budget-usd 1 --price-per-hour 0 >> $LOG 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      if ! on_ac; then
        echo "$(date -u +%FT%TZ) battery: stopping sweep" >> $LOG
        pkill -P $pid; kill $pid; pkill -f "rt.train --cell A-(xs|s)-"; pkill -f "rt.evaluate --cell A-(xs|s)-"
        break
      fi
      sleep 30
    done
    wait $pid 2>/dev/null; rc=$?
    [ $rc -ne 0 ] && { echo "$(date -u +%FT%TZ) sweep exited $rc; pausing 10 min" >> $LOG; sleep 600; }
  done
done
echo "$(date -u +%FT%TZ) MAC SWEEP DONE" >> $LOG
