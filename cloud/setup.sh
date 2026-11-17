#!/usr/bin/env bash
# Runs ON the instance, from ~/rational-transformer. Installs uv, syncs the locked
# environment, runs the tests, then re-runs the learning check and gate on this GPU.
# The sweep refuses to start unless results/gate.json is green at this code hash.
set -euo pipefail
cd "$HOME/rational-transformer"
unset RT_CONFIG RT_EVAL_DIR
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
uv run pytest -q
uv run python -m rt.gate --learn --device cuda
uv run python -m rt.gate
