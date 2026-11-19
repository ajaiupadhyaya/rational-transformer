#!/usr/bin/env bash
# Runs on the Mac. Pulls results every 30 minutes. When the instance writes SWEEP_DONE, does a
# final pull, checks that nothing remains to transfer, and deletes the instance. Also deletes it
# if the instance's run time reaches the hard spend cap, whatever the sweep is doing.
# Usage: cloud/watch.sh <ssh-host> <tnr-id> <price_per_hour> <hard_cap_usd> <created_epoch>
set -uo pipefail
export TNR_NO_SELFUPDATE=1  # tnr 2.0.51 self-update needs sudo and aborts every command
HOST=$1; ID=$2; PRICE=$3; CAP=$4; CREATED=$5
cd "$(git rev-parse --show-toplevel)"
R=rational-transformer/results
# Wall-clock limit for any step (macOS has no `timeout`): a stuck pull can never block the cap.
# Runs the command in its own process group and kills the whole group at the limit, so an
# orphaned rsync cannot keep a pipe open. Exit 124 on timeout.
limit() {
  perl -e '$t=shift; $pid=fork; if(!$pid){setpgrp(0,0); exec @ARGV or exit 127}
    $SIG{ALRM}=sub{kill "TERM",-$pid; sleep 2; kill "KILL",-$pid; exit 124};
    alarm $t; waitpid($pid,0); exit($?>>8)' "$@"
}
SSH="ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o BatchMode=yes"
while true; do
  cost=$(awk -v t0="$CREATED" -v now="$(date +%s)" -v p="$PRICE" 'BEGIN{printf "%.2f", (now-t0)/3600*p}')
  echo "$(date -u +%FT%TZ) cost_est=\$$cost"
  if awk -v c="$cost" -v cap="$CAP" 'BEGIN{exit !(c >= cap)}'; then
    echo "HARD CAP reached; final pull (max 20 min), then deleting instance $ID"
    limit 1200 cloud/pull.sh "$HOST" | tail -2
    tnr delete "$ID" --yes < /dev/null; exit 2
  fi
  limit 1500 cloud/pull.sh "$HOST" | tail -2 || echo "pull failed or timed out (rc=$?); retrying next round"
  if limit 60 $SSH "$HOST" 'test -f rational-transformer/SWEEP_DONE' < /dev/null; then
    limit 60 $SSH "$HOST" 'cat rational-transformer/SWEEP_DONE' < /dev/null
    limit 1500 cloud/pull.sh "$HOST" || echo "final pull failed or timed out"
    left=$(limit 600 rsync -an --timeout=300 -e "$SSH" --checksum --itemize-changes "$HOST:$R/metrics/" results/metrics/ < /dev/null || echo FAIL)
    left=$(printf '%s' "$left" | grep -c . )
    if [ "$left" = "0" ]; then
      echo "SWEEP DONE, metrics verified identical; deleting instance $ID"
      tnr delete "$ID" --yes < /dev/null; tnr status --no-wait < /dev/null | tail -2; exit 0
    fi
    echo "metrics not yet identical ($left items); retrying next round"
  fi
  sleep 1800
done
