#!/usr/bin/env bash
# Runs ON the instance, from ~/rational-transformer. Installs uv, syncs the locked
# environment, runs the tests, then re-runs the learning check and gate on this GPU.
# The sweep refuses to start unless results/gate.json is green at this code hash.
set -euo pipefail
cd "$HOME/rational-transformer"
unset RT_CONFIG RT_EVAL_DIR
# deviation prereg-v1.1: TF32 matmuls and one CPU thread per process on the cloud GPU
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
uv run pytest -q
uv run python -m rt.gate --learn --device cuda
uv run python -m rt.gate
