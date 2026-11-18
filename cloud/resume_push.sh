#!/usr/bin/env bash
# Before relaunching the cloud sweep on a NEW instance: copy finished cloud-cell metrics (so the
# sweep skips them) and checkpoints of cloud cells (so interrupted runs resume from their last
# checkpoint instead of restarting). Usage: cloud/resume_push.sh <ssh-host>
set -euo pipefail
HOST=${1:?usage: cloud/resume_push.sh <ssh-host>}
cd "$(git rev-parse --show-toplevel)"
R=rational-transformer/results
ssh "$HOST" "mkdir -p $R/metrics $R/checkpoints"
rsync -a --include='A-l-*.csv' --include='A-m-*.csv' --include='B-l-*.csv' --exclude='*' results/metrics/ "$HOST:$R/metrics/"
rsync -a --include='A-l-*/***' --include='A-m-*/***' --include='B-l-*/***' --exclude='*' results/checkpoints/ "$HOST:$R/checkpoints/"
ssh "$HOST" "ls $R/metrics | wc -l; ls $R/checkpoints | wc -l"
