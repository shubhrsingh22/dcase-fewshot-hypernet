#!/usr/bin/env bash
set -uo pipefail
cd /mnt/data/dcase-fewshot-hypernet
PY=/mnt/data/.conda_envs/hsed/bin/python
declare -A PL=( [proto]="[]" [hproto1]="[1]" [hproto2]="[2]" [hproto3]="[3]" [hproto_all]="[1,2,3,4]" )
for exp in proto hproto1 hproto2 hproto3 hproto_all; do
  dst="logs/${exp}/thesis_seed42"; mkdir -p "$dst"
  cp -f "logs/$exp/seed_42/best_model.pth" "$dst/"
  echo "=== thesis-subset oracle eval: $exp ==="
  CUDA_VISIBLE_DEVICES=0 $PY -m src.eval_best experiment_name="$exp" \
    model.encoder=protonet model.hyper_placement="${PL[$exp]}" \
    path.eval_dir="$PWD/_val_thesis/Validation_Set" path.feat_eval="$PWD/feat_eval_thesis" \
    path.work_dir="$PWD/$dst" > "logs/console_${exp}_thesis.log" 2>&1
done
echo "ALL THESIS-SUBSET EVALS DONE"
