#!/usr/bin/env bash
# Pilot timing ON the instance (Task 9 step 3). Prints one parseable line per measurement.
#   TRAIN <size> jobs=<J> per_job_steps_s=<...> aggregate_steps_s=<...>
#   EVAL <cell> <cold|warm> seconds=<...>
set -uo pipefail
cd "$HOME/rational-transformer"
unset RT_CONFIG RT_EVAL_DIR
export PATH="$HOME/.local/bin:$PATH"
steps_for() { case $1 in xs|s) echo 1000;; m) echo 600;; l) echo 400;; esac; }
for size in xs s m l; do
  n=$(steps_for $size)
  for j in 1 4 8; do
    tmp=$(mktemp -d)
    for k in $(seq 1 $j); do
      uv run python -m rt.sweep --time-only --cell A-$size-wfull-s0 --steps $n --device cuda \
        > "$tmp/$k.out" 2>&1 &
    done
    wait
    rates=$(awk '/^A-/{print $2}' "$tmp"/*.out | tr '\n' ' ')
    agg=$(echo $rates | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; print s}')
    per=$(echo $rates | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; if(NF) print s/NF; else print "NA"}')
    echo "TRAIN $size jobs=$j per_job_steps_s=$per aggregate_steps_s=$agg rates=[$rates]"
    grep -h -E "Error|Traceback|out of memory" "$tmp"/*.out | head -3
    rm -rf "$tmp"
  done
done
# evaluation cost: size l, 100-step checkpoint; cold then warm benchmark cache
ck=$(mktemp -d)
for cell in A-l-w4-s0 B-l-w4-s0 A-l-wfull-s0; do
  uv run python -m rt.train --cell $cell --device cuda --max-steps 100 --out "$ck" > /dev/null 2>&1
  for pass in cold warm; do
    t0=$(date +%s.%N)
    uv run python -m rt.evaluate --cell $cell --device cuda --ckpt-dir "$ck" --out "$ck/metrics" \
      > "$ck/$cell.$pass.log" 2>&1 || tail -3 "$ck/$cell.$pass.log"
    echo "EVAL $cell $pass seconds=$(echo "$(date +%s.%N) - $t0" | bc)"
  done
done
echo PILOT DONE
