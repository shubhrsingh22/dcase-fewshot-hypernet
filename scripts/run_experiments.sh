#!/usr/bin/env bash
# Run Proto baseline + H-Proto variants across seeds using a per-GPU job queue.
# Features must be extracted once first:  python -m src.train stage.features=true stage.train=false stage.eval=false
#
# Usage:
#   scripts/run_experiments.sh "42 1337 2024 7 2718" "proto hproto1 hproto2 hproto3 hproto_all" "0 1 2 3"
set -uo pipefail

SEEDS=(${1:-42 1337 2024 7 2718})
EXPERIMENTS=(${2:-proto hproto1 hproto2 hproto3 hproto_all})
GPUS=(${3:-0 1 2 3})
PYTHON="${PYTHON:-/mnt/data/.conda_envs/hsed/bin/python}"

# Map experiment name -> hyper_placement override.
placement() {
  case "$1" in
    proto)      echo "[]" ;;
    hproto1)    echo "[1]" ;;
    hproto2)    echo "[2]" ;;
    hproto3)    echo "[3]" ;;
    hproto_all) echo "[1,2,3,4]" ;;
    *)          echo "[]" ;;
  esac
}

JOBS=()
for exp in "${EXPERIMENTS[@]}"; do
  for seed in "${SEEDS[@]}"; do JOBS+=("${exp}:${seed}"); done
done
echo "Queued ${#JOBS[@]} jobs across GPUs: ${GPUS[*]}"
mkdir -p logs

LAST_PID=""
launch() {
  local gpu="$1" exp="${2%%:*}" seed="${2##*:}" pl
  pl="$(placement "$exp")"
  echo ">>> [gpu ${gpu}] ${exp} seed=${seed} placement=${pl}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m src.train \
      experiment_name="${exp}" model.hyper_placement="${pl}" seed="${seed}" \
      path.work_dir="${PWD}/logs/${exp}/seed_${seed}" \
      stage.features=false stage.train=true stage.eval=true \
      || echo "!!! FAILED: ${exp} seed=${seed}"
  ) > "logs/console_${exp}_seed${seed}.log" 2>&1 &
  LAST_PID=$!
}

declare -A GPU_PID
idx=0
for gpu in "${GPUS[@]}"; do
  if (( idx < ${#JOBS[@]} )); then
    launch "$gpu" "${JOBS[$idx]}"; GPU_PID[$gpu]=$LAST_PID; idx=$((idx + 1))
  fi
done

while :; do
  running=0
  for gpu in "${GPUS[@]}"; do
    pid="${GPU_PID[$gpu]:-}"; [ -z "$pid" ] && continue
    if kill -0 "$pid" 2>/dev/null; then
      running=1
    elif (( idx < ${#JOBS[@]} )); then
      launch "$gpu" "${JOBS[$idx]}"; GPU_PID[$gpu]=$LAST_PID; idx=$((idx + 1)); running=1
    else
      GPU_PID[$gpu]=""
    fi
  done
  (( running == 0 )) && break
  sleep 15
done

echo "All runs finished. Aggregate: python -m src.aggregate_results --logs_dir logs --out_dir results"
