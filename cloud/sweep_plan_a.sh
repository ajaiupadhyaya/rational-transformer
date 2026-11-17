#!/usr/bin/env bash
# Runs ON the instance: the Plan A cloud sweep (deviation prereg-v1.1), in priority order,
# under one budget clock. Writes SWEEP_DONE (with each phase's exit code) at the end, always.
# Usage: bash cloud/sweep_plan_a.sh <price_per_hour> <budget_usd> <jobs>
set -uo pipefail
cd "$HOME/rational-transformer"
unset RT_CONFIG RT_EVAL_DIR
export PATH="$HOME/.local/bin:$PATH"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PRICE=${1:?price}; BUDGET=${2:?budget}; JOBS=${3:-3}
T0=$(date +%s)
rm -f SWEEP_DONE
status=""
trap 'echo "phases: $status  finished $(date -u +%FT%TZ)" > SWEEP_DONE' EXIT
for glob in 'A-l-*' 'A-m-*-s[012]' 'B-l-wfull-s[012]' 'B-l-w4-s[012]'; do
  spent=$(awk -v t0="$T0" -v now="$(date +%s)" -v p="$PRICE" 'BEGIN{printf "%.4f", (now-t0)/3600*p}')
  echo "=== $(date -u +%FT%TZ) phase $glob  spent_so_far=\$$spent"
  uv run python -m rt.sweep --device cuda --only "$glob" --jobs "$JOBS" \
    --budget-usd "$BUDGET" --price-per-hour "$PRICE" --already-spent-usd "$spent"
  rc=$?
  status="$status $glob=$rc"
  [ $rc -eq 3 ] && { echo "budget stop"; break; }
done
