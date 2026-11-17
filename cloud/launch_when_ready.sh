#!/usr/bin/env bash
# Runs ON the instance: waits for setup's GATE PASS and the CACHE_READY marker, then runs the
# Plan A sweep. Aborts (writes SWEEP_DONE so the watcher deletes the instance) if setup fails.
cd "$HOME/rational-transformer"
while true; do
  if grep -qE "GATE FAIL|Traceback|failed" setup.log 2>/dev/null; then
    echo "setup failed; not sweeping $(date -u +%FT%TZ)" > SWEEP_DONE; exit 1
  fi
  if grep -q "GATE PASS" setup.log 2>/dev/null && [ -f CACHE_READY ]; then break; fi
  sleep 20
done
exec bash cloud/sweep_plan_a.sh "$@"
