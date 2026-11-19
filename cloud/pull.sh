#!/usr/bin/env bash
# Copies results back from the instance. Never deletes anything local (no --delete).
# Pulls metrics, ledger, logs, gate files, and every per-cell checkpoint + logs (so a lost
# instance's interrupted runs can resume from their last checkpoint).
# Every transfer has a stall timeout (rsync --timeout, ssh keepalives) so a dead connection
# fails within minutes instead of hanging the watcher (2026-11-18: a hung rsync froze it 6 h).
# Usage: cloud/pull.sh <ssh-host>
set -euo pipefail
export RSYNC_RSH="ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o BatchMode=yes"
rsync() { command rsync --timeout=300 "$@" < /dev/null; }
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
rsync -a --include='*/' --include='step*.pt' --include='train_log.csv' --include='run.json' \
  --exclude='*' "$HOST:$R/checkpoints/" results/checkpoints/ 2>/dev/null || echo "(no checkpoints yet)"
echo "metrics files: $(ls results/metrics 2>/dev/null | wc -l | tr -d ' ')"
[ -f results/cloud/ledger.csv ] && echo "cloud ledger rows: $(($(wc -l < results/cloud/ledger.csv) - 1))"
