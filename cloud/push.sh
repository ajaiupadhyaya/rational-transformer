#!/usr/bin/env bash
# Copies the COMMITTED tree (git archive HEAD) to the instance, so the code hash there
# equals the tagged one. Refuses if rt/, configs/ or the lockfile have uncommitted changes.
# Usage: cloud/push.sh <ssh-host>   (the alias tnr writes to ~/.ssh/config on first connect)
set -euo pipefail
HOST=${1:?usage: cloud/push.sh <ssh-host>}
cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain -- rt configs pyproject.toml uv.lock cloud)" ]; then
  echo "uncommitted changes under rt/ configs/ cloud/ or the lockfile; commit first" >&2; exit 1
fi
git archive --format=tar --prefix=rational-transformer/ HEAD | ssh "$HOST" 'tar -xf - -C "$HOME"'
ssh "$HOST" 'cd ~/rational-transformer && echo pushed && ls'
echo "pushed $(git rev-parse --short HEAD) to $HOST"
