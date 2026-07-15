#!/usr/bin/env bash
# Wait for the 4-conv ProtoNet + Surrey-contrastive run to finish, score it with
# the oracle best-operating-point eval, compare against the plain ProtoNet, and
# if the contrastive technique improves F, run the H-Proto + contrastive variant.
set -uo pipefail
cd /mnt/data/dcase-fewshot-hypernet
PY=/mnt/data/.conda_envs/hsed/bin/python
GPU=0
WAIT_PID="${1:?need proto_surrey training PID}"

echo "[orch] waiting for proto_surrey PID ${WAIT_PID} ..."
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 30; done
echo "[orch] proto_surrey training done."

echo "[orch] oracle eval: proto_surrey"
CUDA_VISIBLE_DEVICES=$GPU $PY -m src.eval_best experiment_name=proto_surrey \
  model.encoder=protonet path.work_dir="$PWD/logs/proto_surrey/seed_42"

echo "[orch] oracle eval: plain proto (reference)"
CUDA_VISIBLE_DEVICES=$GPU $PY -m src.eval_best experiment_name=proto \
  model.encoder=protonet path.work_dir="$PWD/logs/proto/seed_42"

F_S=$($PY -c "import json;print(json.load(open('logs/proto_surrey/seed_42/results_best.json'))['overall']['fmeasure (percentage)'])")
F_P=$($PY -c "import json;print(json.load(open('logs/proto/seed_42/results_best.json'))['overall']['fmeasure (percentage)'])")
echo "[orch] proto_surrey best F=${F_S} | plain proto best F=${F_P}"

IMPROVED=$($PY -c "print(1 if ${F_S} > ${F_P} else 0)")
if [ "${IMPROVED}" != "1" ]; then
  echo "[orch] contrastive did NOT improve over plain proto; skipping H-Proto."
  exit 0
fi

echo "[orch] contrastive improved -> running H-Proto-3 + contrastive"
CUDA_VISIBLE_DEVICES=$GPU $PY -m src.train \
  path.work_dir="$PWD/logs/hproto3_surrey/seed_42" \
  stage.features=false stage.train=true stage.eval=true \
  experiment_name=hproto3_surrey model.encoder=protonet model.hyper_placement="[3]" \
  train.contrast=true train.epochs=50 train.patience=10 \
  > logs/console_hproto3_surrey_seed42.log 2>&1

echo "[orch] oracle eval: hproto3_surrey"
CUDA_VISIBLE_DEVICES=$GPU $PY -m src.eval_best experiment_name=hproto3_surrey \
  model.encoder=protonet model.hyper_placement="[3]" \
  path.work_dir="$PWD/logs/hproto3_surrey/seed_42"

F_H=$($PY -c "import json;print(json.load(open('logs/hproto3_surrey/seed_42/results_best.json'))['overall']['fmeasure (percentage)'])")
echo "[orch] DONE. plain proto=${F_P}  proto_surrey=${F_S}  hproto3_surrey=${F_H}"
