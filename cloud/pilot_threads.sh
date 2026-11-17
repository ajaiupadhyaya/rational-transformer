#!/usr/bin/env bash
# Second pilot pass: windowed training speed, one-thread CPU runs, TF32 (env override only).
set -uo pipefail
cd "$HOME/rational-transformer"
unset RT_CONFIG RT_EVAL_DIR
export PATH="$HOME/.local/bin:$PATH"
run() {  # $1 cell, $2 steps, $3 jobs, $4 tag
  tmp=$(mktemp -d)
  for k in $(seq 1 $3); do
    uv run python -m rt.sweep --time-only --cell $1 --steps $2 --device cuda > "$tmp/$k.out" 2>&1 &
  done
  wait
  rates=$(awk '/^[AB]-/{print $2}' "$tmp"/*.out | tr '\n' ' ')
  agg=$(echo $rates | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; print s}')
  echo "$4 $1 jobs=$3 aggregate_steps_s=$agg rates=[$rates]"
  grep -h -E "Error|out of memory" "$tmp"/*.out | head -2
  rm -rf "$tmp"
}
for c in A-l-w16-s0 A-l-w2-s0 A-m-w16-s0; do
  for j in 1 4; do run $c 1000 $j W; done
done
for c in A-xs-wfull-s0 A-s-wfull-s0; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 run $c 1000 8 T1
done
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
uv run python -c "import torch; print('TF32 cublas allowed:', torch.backends.cuda.matmul.allow_tf32)"
run A-l-wfull-s0 400 1 TF32
run A-l-w16-s0 1000 1 TF32
run A-m-wfull-s0 600 1 TF32
ck=/tmp/tmp.7aoUxic99r
uv run python -m rt.evaluate --cell A-l-w4-s0 --device cuda --ckpt-dir $ck --out /tmp/tf32_metrics 2>&1 | grep -E "converged=|Error" | sed 's/^/TF32EVAL /'
echo THREADS DONE
