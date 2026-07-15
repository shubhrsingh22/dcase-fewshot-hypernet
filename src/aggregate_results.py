"""Aggregate per-run results.json into thesis Table 3.3 (Proto vs H-Proto).

Reports precision / recall / F-measure as mean +/- 95% CI over trials (seeds),
matching the thesis presentation.

    python -m src.aggregate_results --logs_dir logs --out_dir results
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from statistics import mean, stdev

# Display order and labels for the experiments.
ORDER = ["random", "proto", "hproto1", "hproto2", "hproto3", "hproto_all"]
LABELS = {
    "random": "Random baseline",
    "proto": "Proto Network", "hproto1": "H-Proto-1", "hproto2": "H-Proto-2",
    "hproto3": "H-Proto-3", "hproto_all": "H-Proto-All",
}


def ci95(vals):
    if len(vals) < 2:
        return 0.0
    return 1.96 * stdev(vals) / (len(vals) ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="logs")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    by_exp = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(args.logs_dir, "**", "results.json"), recursive=True):
        with open(path) as f:
            r = json.load(f)
        o = r["overall"]
        exp = r["experiment_name"]
        by_exp[exp]["precision"].append(o["precision"] * 100)
        by_exp[exp]["recall"].append(o["recall"] * 100)
        # F-measure is already stored as a percentage by the official evaluator.
        by_exp[exp]["f"].append(o["fmeasure (percentage)"])
        by_exp[exp]["seeds"].append(r.get("seed"))

    exps = [e for e in ORDER if e in by_exp] + [e for e in by_exp if e not in ORDER]
    rows = []
    for exp in exps:
        d = by_exp[exp]
        rows.append({
            "experiment": LABELS.get(exp, exp),
            "n_trials": len(d["f"]),
            "seeds": d["seeds"],
            "precision": mean(d["precision"]), "precision_ci": ci95(d["precision"]),
            "recall": mean(d["recall"]), "recall_ci": ci95(d["recall"]),
            "f": mean(d["f"]), "f_ci": ci95(d["f"]),
        })

    csv_path = os.path.join(args.out_dir, "table3_3.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "n_trials", "Precision(%)", "Recall(%)", "F-score(%)"])
        for r in rows:
            w.writerow([r["experiment"], r["n_trials"],
                        f"{r['precision']:.2f}+-{r['precision_ci']:.2f}",
                        f"{r['recall']:.2f}+-{r['recall_ci']:.2f}",
                        f"{r['f']:.2f}+-{r['f_ci']:.2f}"])

    md_path = os.path.join(args.out_dir, "table3_3.md")
    with open(md_path, "w") as f:
        f.write("# Table 3.3 — DCASE few-shot bioacoustic detection\n\n")
        f.write("Mean ± 95% CI over trials on the DCASE Validation Set "
                "(harmonic mean of per-subset scores).\n\n")
        f.write("| Model | Trials | Precision (%) | Recall (%) | F-score (%) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['experiment']} | {r['n_trials']} | "
                    f"{r['precision']:.2f} ± {r['precision_ci']:.2f} | "
                    f"{r['recall']:.2f} ± {r['recall_ci']:.2f} | "
                    f"{r['f']:.2f} ± {r['f_ci']:.2f} |\n")

    print(f"Wrote {csv_path} and {md_path}\n")
    for r in rows:
        print(f"{r['experiment']:16s} F={r['f']:.2f}±{r['f_ci']:.2f}  "
              f"P={r['precision']:.2f}  R={r['recall']:.2f}  (n={r['n_trials']})")


if __name__ == "__main__":
    main()
