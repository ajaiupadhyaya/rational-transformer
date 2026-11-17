#!/usr/bin/env bash
# Copies the benchmark cache built on the Mac (results/eval) to the instance, skipping
# Family B windows the Plan A sweep does not run. Usage: cloud/upload_cache.sh <ssh-host>
set -euo pipefail
HOST=${1:?usage: cloud/upload_cache.sh <ssh-host>}
cd "$(git rev-parse --show-toplevel)"
ssh "$HOST" 'mkdir -p rational-transformer/results/eval'
rsync -a --info=progress2 --exclude='B_*_bayes-w2.npy' --exclude='B_*_bayes-w8.npy' \
  --exclude='B_*_bayes-w16.npy' results/eval/ "$HOST:rational-transformer/results/eval/"
