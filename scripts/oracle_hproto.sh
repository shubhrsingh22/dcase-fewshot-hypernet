#!/usr/bin/env bash
set -uo pipefail
cd /mnt/data/dcase-fewshot-hypernet
PY=/mnt/data/.conda_envs/hsed/bin/python
declare -A PL=( [hproto1]="[1]" [hproto2]="[2]" [hproto3]="[3]" [hproto_all]="[1,2,3,4]" )
for exp in hproto1 hproto2 hproto3 hproto_all; do
  echo "=== oracle eval: $exp seed 42 ==="
  CUDA_VISIBLE_DEVICES=0 $PY -m src.eval_best experiment_name=$exp \
    model.encoder=protonet model.hyper_placement="${PL[$exp]}" \
    path.work_dir="$PWD/logs/$exp/seed_42"
done
echo "ALL ORACLE EVALS DONE"
