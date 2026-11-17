#!/usr/bin/env bash
# Copies results back from the instance. Never deletes anything local (no --delete).
# Pulls metrics, ledger, logs, gate files, and per-cell final checkpoint + logs.
# Usage: cloud/pull.sh <ssh-host>
set -euo pipefail
HOST=${1:?usage: cloud/pull.sh <ssh-host>}
cd "$(git rev-parse --show-toplevel)"
R=rational-transformer/results
mkdir -p results/checkpoints
mkdir -p results/cloud
for p in metrics logs; do
  rsync -a "$HOST:$R/$p/" "results/$p/" 2>/dev/null || echo "(no $p yet)"
done
# the cloud ledger sits beside the Mac ledger (results/ledger.csv), never over it
rsync -a "$HOST:$R/ledger.csv" results/cloud/ledger.csv 2>/dev/null || echo "(no ledger yet)"
rsync -a "$HOST:rational-transformer/sweep.log" results/cloud/sweep.log 2>/dev/null || true
# the instance's own gate evidence goes beside, never over, the local gate files
rsync -a "$HOST:$R/gate.json" results/cloud/gate.json 2>/dev/null || echo "(no gate.json yet)"
rsync -a "$HOST:$R/gate/" results/cloud/gate/ 2>/dev/null || echo "(no gate dir yet)"
rsync -a --include='*/' --include='step30000.pt' --include='train_log.csv' --include='run.json' \
  --exclude='*' "$HOST:$R/checkpoints/" results/checkpoints/ 2>/dev/null || echo "(no checkpoints yet)"
echo "metrics files: $(ls results/metrics 2>/dev/null | wc -l | tr -d ' ')"
[ -f results/cloud/ledger.csv ] && echo "cloud ledger rows: $(($(wc -l < results/cloud/ledger.csv) - 1))"
