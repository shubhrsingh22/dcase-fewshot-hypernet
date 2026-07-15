#!/usr/bin/env bash
# Overnight run: finish current oracle/energy evals, then run energy-negative
# oracle eval for every H-Proto variant, and write a final comparison table.
set -uo pipefail
cd /mnt/data/dcase-fewshot-hypernet
PY=/mnt/data/.conda_envs/hsed/bin/python
ENERGY_PID="${1:-}"; ORACLE_PID="${2:-}"

echo "[night] waiting for in-flight jobs (${ENERGY_PID}, ${ORACLE_PID}) ..."
for pid in "$ENERGY_PID" "$ORACLE_PID"; do
  [ -n "$pid" ] && while kill -0 "$pid" 2>/dev/null; do sleep 30; done
done
echo "[night] in-flight jobs done."

declare -A PL=( [hproto1]="[1]" [hproto2]="[2]" [hproto3]="[3]" [hproto_all]="[1,2,3,4]" )

# Energy-negative oracle eval for each H-Proto variant (GPU 1).
for exp in hproto1 hproto2 hproto3 hproto_all; do
  dst="logs/${exp}_energy/seed_42"
  mkdir -p "$dst"
  cp -f "logs/$exp/seed_42/best_model.pth" "$dst/" 2>/dev/null || { echo "[night] missing $exp model"; continue; }
  echo "[night] energy oracle eval: $exp"
  CUDA_VISIBLE_DEVICES=1 $PY -m src.eval_best experiment_name="$exp" \
    model.encoder=protonet model.hyper_placement="${PL[$exp]}" \
    path.work_dir="$PWD/$dst" eval.neg_estimate=energy \
    > "logs/console_${exp}_energy.log" 2>&1 || echo "[night] FAILED $exp energy"
done

echo "[night] building final summary ..."
$PY - <<'PYEOF'
import json, os
def load(p):
    try:
        return json.load(open(p))["overall"]["fmeasure (percentage)"]
    except Exception:
        return None
rows = [
    ("Proto (baseline)", "proto", "proto_energy"),
    ("H-Proto-1", "hproto1", "hproto1_energy"),
    ("H-Proto-2", "hproto2", "hproto2_energy"),
    ("H-Proto-3", "hproto3", "hproto3_energy"),
    ("H-Proto-All", "hproto_all", "hproto_all_energy"),
]
lines = ["# Oracle best-operating-point eval (seed 42) — random vs energy negatives\n",
         "| Model | F (random neg) | F (energy neg) |", "|---|---|---|"]
for name, rdir, edir in rows:
    fr = load(f"logs/{rdir}/seed_42/results_best.json")
    fe = load(f"logs/{edir}/seed_42/results_best.json")
    fr = f"{fr:.2f}" if fr is not None else "-"
    fe = f"{fe:.2f}" if fe is not None else "-"
    lines.append(f"| {name} | {fr} | {fe} |")
os.makedirs("results", exist_ok=True)
open("results/oracle_summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PYEOF
echo "[night] DONE. Summary at results/oracle_summary.md"
