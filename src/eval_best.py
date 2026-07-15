"""Best-operating-point evaluation (Surrey DCASE-2024 reporting protocol).

Caches per-recording probabilities once, then sweeps the probability threshold
(0.05 steps) and both post-processors (adaptive fraction + fixed min-duration),
reporting the best overall event-based F-measure. This mirrors how the Surrey
system selects/reports its validation number.

    python -m src.eval_best experiment_name=proto_surrey path.work_dir=logs/proto_surrey/seed_42
"""

import contextlib
import io
import json
import os
import sys
from glob import glob

import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_encoder
from src.post_proc import run_post_processing
from src.util import compute_probs, probs_to_events, energy_negative_mask, eval_file_meta
from evaluation.evaluation import evaluate


def main():
    base = OmegaConf.load(os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))
    conf = OmegaConf.merge(base, OmegaConf.from_dotlist(sys.argv[1:]))
    OmegaConf.resolve(conf)
    device = torch.device(conf.device if torch.cuda.is_available() else "cpu")

    encoder = build_encoder(conf)
    state = torch.load(os.path.join(conf.path.work_dir, "best_model.pth"), map_location=device)
    encoder.load_state_dict(state["encoder"]); encoder.to(device)

    cache = []
    for feat_file in sorted(glob(os.path.join(conf.path.feat_eval, "*.h5"))):
        audio = os.path.basename(feat_file).replace(".h5", ".wav")
        name = os.path.basename(feat_file).replace(".h5", "")
        neg_mask = None
        if conf.eval.get("neg_estimate", "random") == "energy":
            ap, pf = eval_file_meta(conf, name)
            if ap is not None:
                neg_mask = energy_negative_mask(conf, ap, pf)
        with h5py.File(feat_file, "r") as hf:
            strt = hf["start_index_query"][:][0]
            prob, hop_seg = compute_probs(conf, hf, encoder, device, neg_mask=neg_mask)
        cache.append((audio, prob, hop_seg, strt))
        print(f"cached {audio}: {len(prob)} probs", flush=True)

    def score(thresh, mode, param):
        names, ons, offs = [], [], []
        for audio, prob, hop_seg, strt in cache:
            on, off = probs_to_events(prob, thresh, hop_seg, strt, conf)
            names.extend([audio] * len(on)); ons.extend(on); offs.extend(off)
        raw = os.path.join(conf.path.work_dir, "_best_raw.csv")
        pp = os.path.join(conf.path.work_dir, "_best_pp.csv")
        pd.DataFrame({"Audiofilename": names, "Starttime": ons, "Endtime": offs}).to_csv(raw, index=False)
        run_post_processing(mode, conf.path.eval_dir, raw, pp, n_shots=conf.train.n_shot,
                            min_dur=param if mode == "fixed" else 0.2,
                            adaptive_frac=param if mode == "adaptive" else 0.6)
        with contextlib.redirect_stdout(io.StringIO()):
            ov = evaluate(pp, conf.path.eval_dir, "best", "VAL", conf.path.work_dir)
        return ov

    best = None
    for thresh in np.arange(0.05, 0.95, 0.05):
        for mode, param in ([("adaptive", f) for f in (0.2, 0.3, 0.4, 0.5, 0.6)] +
                            [("fixed", m) for m in (0.05, 0.1, 0.15, 0.2)]):
            ov = score(round(float(thresh), 2), mode, param)
            f = ov["fmeasure (percentage)"]
            if best is None or f > best["f"]:
                best = {"f": f, "precision": ov["precision"] * 100, "recall": ov["recall"] * 100,
                        "threshold": round(float(thresh), 2), "post_proc": mode, "param": param}
        print(f"thresh={thresh:.2f} best-so-far F={best['f']:.2f} "
              f"({best['post_proc']} {best['param']}, thr {best['threshold']})", flush=True)

    print(f"\nBEST: F={best['f']:.2f}  P={best['precision']:.2f}  R={best['recall']:.2f}  "
          f"thr={best['threshold']} post={best['post_proc']}({best['param']})")
    result = {"experiment_name": conf.experiment_name, "seed": conf.seed,
              "overall": {"precision": best["precision"] / 100, "recall": best["recall"] / 100,
                          "fmeasure (percentage)": best["f"]}, "best_config": best}
    with open(os.path.join(conf.path.work_dir, "results_best.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
