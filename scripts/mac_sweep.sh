#!/usr/bin/env bash
# Plan A (prereg-v1.1): sizes xs and s on the Mac (mps), 2 jobs, low priority. Runs the sweep
# only on mains power: kills it on battery, restarts it on AC (cells resume from their last
# checkpoint). Exits when both phases report nothing left to run.
set -uo pipefail
# Keep the Mac from idle- or system-sleeping while the runner is alive (2026-11-18: the Mac
# slept through the whole 01:00-08:00 window). -s applies only on AC; closing the lid still sleeps.
if [ -z "${RT_CAFFEINATED:-}" ]; then RT_CAFFEINATED=1 exec caffeinate -i -s "$0" "$@"; fi
cd "$(git rev-parse --show-toplevel)"
unset RT_CONFIG RT_EVAL_DIR
LOG=results/mac_sweep.log
# ALLOW_BATTERY_FLOOR=N (owner's one-off permission): also run on battery while charge > N%.
# RUN_HOURS="1-7" (owner, 2026-11-18): heavy local compute only while the owner sleeps,
# i.e. local hours 01:00-07:59. Outside the window the runner waits (or stops a running sweep).
in_window() {
  [ -n "${RUN_HOURS:-}" ] || return 0
  h=$((10#$(date +%H))); lo=${RUN_HOURS%-*}; hi=${RUN_HOURS#*-}
  [ "$h" -ge "$lo" ] && [ "$h" -le "$hi" ]
}
on_ac() {
  in_window || return 1
  pmset -g batt | grep -q "AC Power" && return 0
  [ -n "${ALLOW_BATTERY_FLOOR:-}" ] || return 1
  pct=$(pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d %)
  [ "${pct:-0}" -gt "$ALLOW_BATTERY_FLOOR" ]
}
remaining() { uv run python -m rt.sweep --device mps --only "$1" --dry-run 2>/dev/null | grep -c '^A-'; }
for glob in 'A-xs-*' 'A-s-*'; do
  while [ "$(remaining "$glob")" -gt 0 ]; do
    if ! on_ac; then echo "$(date -u +%FT%TZ) waiting (battery or outside RUN_HOURS)" >> $LOG; sleep 600; continue; fi
    echo "$(date -u +%FT%TZ) start $glob" >> $LOG
    nice -n 15 uv run python -m rt.sweep --device mps --only "$glob" --jobs 2 --budget-usd 1 --price-per-hour 0 >> $LOG 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      if ! on_ac; then
        echo "$(date -u +%FT%TZ) battery or outside RUN_HOURS: stopping sweep" >> $LOG
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
