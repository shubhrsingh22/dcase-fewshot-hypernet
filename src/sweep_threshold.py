"""Sweep the detection threshold for a trained model without re-running the encoder.

Computes the averaged per-segment probabilities once per validation recording
(the expensive step), caches them, then evaluates the event-based F-measure for a
list of thresholds. Useful for picking the operating point.

    python -m src.sweep_threshold experiment_name=proto \
        path.work_dir=logs/proto/seed_42 thresholds="0.5,0.6,0.7,0.8"
"""

import contextlib
import io
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
from src.util import compute_probs, probs_to_events
from evaluation.evaluation import evaluate


def main():
    base = OmegaConf.load(os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))
    cli = OmegaConf.from_dotlist([a for a in sys.argv[1:] if not a.startswith("thresholds=")])
    conf = OmegaConf.merge(base, cli)
    OmegaConf.resolve(conf)
    thr_arg = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("thresholds=")), "0.5,0.6,0.7,0.8,0.9")
    thresholds = [float(t) for t in thr_arg.split(",")]

    device = torch.device(conf.device if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(conf)
    state = torch.load(os.path.join(conf.path.work_dir, "best_model.pth"), map_location=device)
    encoder.load_state_dict(state["encoder"])
    encoder.to(device)

    # Cache probabilities per file once.
    cache = []
    for feat_file in sorted(glob(os.path.join(conf.path.feat_eval, "*.h5"))):
        audio_name = os.path.basename(feat_file).replace(".h5", ".wav")
        with h5py.File(feat_file, "r") as hf:
            strt = hf["start_index_query"][:][0]
            prob, hop_seg = compute_probs(conf, hf, encoder, device)
        cache.append((audio_name, prob, hop_seg, strt))
        print(f"cached {audio_name}: {len(prob)} query probs", flush=True)

    def score(thresh):
        names, ons, offs = [], [], []
        for audio_name, prob, hop_seg, strt in cache:
            on, off = probs_to_events(prob, thresh, hop_seg, strt, conf)
            names.extend([audio_name] * len(on)); ons.extend(on); offs.extend(off)
        raw = os.path.join(conf.path.work_dir, f"_sweep_raw_{thresh}.csv")
        pp = os.path.join(conf.path.work_dir, f"_sweep_pp_{thresh}.csv")
        pd.DataFrame({"Audiofilename": names, "Starttime": ons, "Endtime": offs}).to_csv(raw, index=False)
        run_post_processing(conf.eval.get("post_proc", "fixed"), conf.path.eval_dir, raw, pp,
                            n_shots=conf.train.n_shot, min_dur=conf.eval.get("min_dur", 0.2),
                            adaptive_frac=conf.eval.get("adaptive_frac", 0.6))
        with contextlib.redirect_stdout(io.StringIO()):
            ov = evaluate(pp, conf.path.eval_dir, "sweep", "VAL", conf.path.work_dir)
        return ov

    print(f"\n{'thresh':>7s} {'P%':>7s} {'R%':>7s} {'F%':>7s}")
    for t in thresholds:
        ov = score(t)
        print(f"{t:7.2f} {ov['precision']*100:7.2f} {ov['recall']*100:7.2f} {ov['fmeasure (percentage)']:7.2f}", flush=True)


if __name__ == "__main__":
    main()
