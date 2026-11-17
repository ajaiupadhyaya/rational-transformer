#!/usr/bin/env bash
# Runs on the Mac. Pulls results every 30 minutes. When the instance writes SWEEP_DONE, does a
# final pull, checks that nothing remains to transfer, and deletes the instance. Also deletes it
# if the instance's run time reaches the hard spend cap, whatever the sweep is doing.
# Usage: cloud/watch.sh <ssh-host> <tnr-id> <price_per_hour> <hard_cap_usd> <created_epoch>
set -uo pipefail
HOST=$1; ID=$2; PRICE=$3; CAP=$4; CREATED=$5
cd "$(git rev-parse --show-toplevel)"
R=rational-transformer/results
while true; do
  cost=$(awk -v t0="$CREATED" -v now="$(date +%s)" -v p="$PRICE" 'BEGIN{printf "%.2f", (now-t0)/3600*p}')
  echo "$(date -u +%FT%TZ) cost_est=\$$cost"
  cloud/pull.sh "$HOST" | tail -2
  if awk -v c="$cost" -v cap="$CAP" 'BEGIN{exit !(c >= cap)}'; then
    echo "HARD CAP reached; deleting instance $ID"; tnr delete "$ID" --yes < /dev/null; exit 2
  fi
  if ssh -o ConnectTimeout=15 "$HOST" 'test -f rational-transformer/SWEEP_DONE' < /dev/null; then
    ssh "$HOST" 'cat rational-transformer/SWEEP_DONE' < /dev/null
    cloud/pull.sh "$HOST"
    left=$(rsync -an --checksum --itemize-changes "$HOST:$R/metrics/" results/metrics/ | wc -l | tr -d ' ')
    if [ "$left" = "0" ]; then
      echo "SWEEP DONE, metrics verified identical; deleting instance $ID"
      tnr delete "$ID" --yes < /dev/null; tnr status --no-wait < /dev/null | tail -2; exit 0
    fi
    echo "metrics not yet identical ($left items); retrying next round"
  fi
  sleep 1800
done
