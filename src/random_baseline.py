"""Random-guessing baseline for the few-shot detection task (thesis Table 3.3).

Required by the examiner correction report ("Add random guessing baseline...").
For each validation recording it assigns every query segment an i.i.d. uniform
random positive-class probability, then runs the *identical* event-formation,
post-processing and event-based scoring used by the trained models. Averaged
over seeds this gives the chance-level floor for the metric.

    python -m src.random_baseline seed=42 path.work_dir=logs/random/seed_42
"""

import json
import os
import sys
from glob import glob

import h5py
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.post_proc import run_post_processing
from src.util import probs_to_events
from evaluation.evaluation import evaluate


def main():
    base = OmegaConf.load(os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))
    conf = OmegaConf.merge(base, OmegaConf.from_dotlist(sys.argv[1:]))
    OmegaConf.resolve(conf)
    conf.experiment_name = "random"
    os.makedirs(conf.path.work_dir, exist_ok=True)
    rng = np.random.default_rng(conf.seed)

    names, ons, offs = [], [], []
    for feat_file in sorted(glob(os.path.join(conf.path.feat_eval, "*.h5"))):
        audio_name = os.path.basename(feat_file).replace(".h5", ".wav")
        with h5py.File(feat_file, "r") as hf:
            n_query = hf["feat_query"].shape[0]
            hop_seg = hf["hop_seg"][:]
            strt = hf["start_index_query"][:][0]
        prob = rng.random(n_query)  # uniform(0,1) per query segment
        on, off = probs_to_events(prob, conf.eval.threshold, hop_seg, strt, conf)
        names.extend([audio_name] * len(on)); ons.extend(on); offs.extend(off)

    raw = os.path.join(conf.path.work_dir, "Eval_out_raw.csv")
    pp = os.path.join(conf.path.work_dir, "Eval_out_postproc.csv")
    pd.DataFrame({"Audiofilename": names, "Starttime": ons, "Endtime": offs}).to_csv(raw, index=False)
    run_post_processing(conf.eval.get("post_proc", "fixed"), conf.path.eval_dir, raw, pp,
                        n_shots=conf.train.n_shot, min_dur=conf.eval.get("min_dur", 0.2),
                        adaptive_frac=conf.eval.get("adaptive_frac", 0.6))
    overall = evaluate(pp, conf.path.eval_dir, "random", "VAL", conf.path.work_dir)
    result = {"experiment_name": "random", "seed": conf.seed,
              "hyper_placement": [], "overall": {k: float(v) for k, v in overall.items()}}
    with open(os.path.join(conf.path.work_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[random] seed={conf.seed} -> {overall}", flush=True)


if __name__ == "__main__":
    main()
