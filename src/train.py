"""Entry point: feature extraction / episodic training / evaluation.

Reproduces the thesis Chapter 3.3 pipeline (ProtoNet baseline + H-Proto variants)
on the DCASE few-shot bioacoustic Development Set. Config is loaded with OmegaConf
and can be overridden from the CLI with dotlist syntax, e.g.

    python -m src.train stage.features=true stage.train=false stage.eval=false
    python -m src.train experiment_name=hproto3 model.hyper_placement=[3] seed=42
"""

import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.batch_sampler import EpisodicBatchSampler
from src.datagen import Datagen
from src.feature_extract import extract_eval, extract_train
from src.model import build_encoder
from src.post_proc import run_post_processing
from src.util import (evaluate_prototypes, prototypical_loss, prototypical_loss_filter_negative,
                      energy_negative_mask, eval_file_meta)
import h5py
from glob import glob
from evaluation.evaluation import evaluate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_protonet(encoder, train_loader, valid_loader, conf, n_tr, n_vd):
    device = torch.device(conf.device if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    optim = torch.optim.Adam(encoder.parameters(), lr=conf.train.lr_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, gamma=conf.train.scheduler_gamma, step_size=conf.train.scheduler_step_size
    )

    best_val_acc, best_epoch = 0.0, -1
    os.makedirs(conf.path.work_dir, exist_ok=True)
    best_model_path = os.path.join(conf.path.work_dir, "best_model.pth")

    for epoch in range(conf.train.epochs):
        encoder.train()
        tr_loss, tr_acc = [], []
        for x, y in train_loader:
            optim.zero_grad()
            out = encoder(x.to(device))
            loss, acc = prototypical_loss(out, y.to(device), conf.train.n_shot)
            loss.backward()
            optim.step()
            tr_loss.append(loss.item()); tr_acc.append(acc.item())
        scheduler.step()

        encoder.eval()
        vd_loss, vd_acc = [], []
        with torch.no_grad():
            for x, y in valid_loader:
                out = encoder(x.to(device))
                loss, acc = prototypical_loss(out, y.to(device), conf.train.n_shot)
                vd_loss.append(loss.item()); vd_acc.append(acc.item())

        avg_vd_acc = float(np.mean(vd_acc))
        print(f"Epoch {epoch:03d} | train loss {np.mean(tr_loss):.4f} acc {np.mean(tr_acc):.4f}"
              f" | val loss {np.mean(vd_loss):.4f} acc {avg_vd_acc:.4f}", flush=True)

        if avg_vd_acc > best_val_acc:
            best_val_acc, best_epoch = avg_vd_acc, epoch
            torch.save({"encoder": encoder.state_dict()}, best_model_path)
        elif epoch - best_epoch >= conf.train.patience:
            print(f"Early stopping at epoch {epoch} (best {best_val_acc:.4f} @ {best_epoch})")
            break

    return best_val_acc


def run_train(conf):
    set_seed(conf.seed)
    gen = Datagen(conf)
    X_train, Y_train, X_val, Y_val = gen.generate_train()
    print(f"[train] {len(Y_train)} train / {len(Y_val)} val segments, "
          f"{gen.n_classes} classes", flush=True)

    samples_per_cls = conf.train.n_shot * 2
    n_ep = conf.train.num_episodes
    sampler_tr = EpisodicBatchSampler(Y_train, n_ep, conf.train.k_way, samples_per_cls)
    sampler_vd = EpisodicBatchSampler(Y_val, n_ep, conf.train.k_way, samples_per_cls)
    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.LongTensor(Y_train))
    valid_ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.LongTensor(Y_val))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_sampler=sampler_tr, num_workers=4, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_sampler=sampler_vd, num_workers=4, pin_memory=True)

    encoder = build_encoder(conf)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[model] {conf.experiment_name} params={n_params/1e6:.3f}M "
          f"hyper_placement={list(conf.model.hyper_placement)}", flush=True)
    best_acc = train_protonet(encoder, train_loader, valid_loader, conf, n_ep, n_ep)
    print(f"[train] best val accuracy {best_acc:.4f}", flush=True)


def run_train_contrastive(conf):
    """Surrey-style hard-negative contrastive training on the chosen encoder."""
    from src.contrastive_dataset import ContrastiveDynamicDataset, IdentityBatchSampler
    from src.datagen import Datagen

    set_seed(conf.seed)
    device = torch.device(conf.device if torch.cuda.is_available() else "cpu")
    stats = Datagen(conf)  # reuse the exact training-set mean/std used at eval
    print(f"[contrast] building dynamic dataset (feature cache) ...", flush=True)
    ds = ContrastiveDynamicDataset(conf, stats.mean, stats.std)
    print(f"[contrast] {len(ds.classes)} classes cached", flush=True)

    spc = conf.train.n_shot * 2
    n_ep = conf.train.num_episodes
    tr_sampler = IdentityBatchSampler(len(ds.classes), conf.train.k_way, spc, n_ep)
    vd_sampler = IdentityBatchSampler(len(ds.classes), conf.train.k_way, spc, max(50, n_ep // 5))
    tr_loader = torch.utils.data.DataLoader(ds, batch_sampler=tr_sampler, num_workers=0)
    vd_loader = torch.utils.data.DataLoader(ds, batch_sampler=vd_sampler, num_workers=0)

    encoder = build_encoder(conf).to(device)
    print(f"[model] {conf.experiment_name} params={sum(p.numel() for p in encoder.parameters())/1e6:.3f}M "
          f"contrastive hyper={list(conf.model.hyper_placement)}", flush=True)
    optim = torch.optim.Adam(encoder.parameters(), lr=conf.train.lr_rate)
    sched = torch.optim.lr_scheduler.StepLR(optim, gamma=conf.train.scheduler_gamma,
                                            step_size=conf.train.scheduler_step_size)
    os.makedirs(conf.path.work_dir, exist_ok=True)
    best_path = os.path.join(conf.path.work_dir, "best_model.pth")
    best_acc, best_epoch = 0.0, -1

    for epoch in range(conf.train.epochs):
        encoder.train(); tr_l, tr_a = [], []
        for pos, neg, y, y_neg in tr_loader:
            optim.zero_grad()
            x = torch.cat([pos, neg], 0).to(device)
            yy = torch.cat([y, y_neg], 0).to(device)
            out = encoder(x)
            loss, acc = prototypical_loss_filter_negative(out, yy, conf.train.n_shot)
            loss.backward(); optim.step()
            tr_l.append(loss.item()); tr_a.append(acc.item())
        sched.step()
        encoder.eval(); vd_a = []
        with torch.no_grad():
            for pos, neg, y, y_neg in vd_loader:
                x = torch.cat([pos, neg], 0).to(device)
                yy = torch.cat([y, y_neg], 0).to(device)
                loss, acc = prototypical_loss_filter_negative(encoder(x), yy, conf.train.n_shot)
                vd_a.append(acc.item())
        va = float(np.mean(vd_a))
        print(f"Epoch {epoch:03d} | train loss {np.mean(tr_l):.4f} acc {np.mean(tr_a):.4f} | val acc {va:.4f}", flush=True)
        if va > best_acc:
            best_acc, best_epoch = va, epoch
            torch.save({"encoder": encoder.state_dict()}, best_path)
        elif epoch - best_epoch >= conf.train.patience:
            print(f"Early stopping at epoch {epoch} (best {best_acc:.4f} @ {best_epoch})"); break
    print(f"[contrast] best val acc {best_acc:.4f}", flush=True)


def run_eval(conf):
    device = torch.device(conf.device if torch.cuda.is_available() else "cpu")
    set_seed(conf.seed)
    encoder = build_encoder(conf)
    state = torch.load(os.path.join(conf.path.work_dir, "best_model.pth"), map_location=device)
    encoder.load_state_dict(state["encoder"])
    encoder.to(device)

    name_arr, onset_arr, offset_arr = [], [], []
    for feat_file in sorted(glob(os.path.join(conf.path.feat_eval, "*.h5"))):
        audio_name = os.path.basename(feat_file).replace(".h5", ".wav")
        name = os.path.basename(feat_file).replace(".h5", "")
        neg_mask = None
        if conf.eval.get("neg_estimate", "random") == "energy":
            ap, pf = eval_file_meta(conf, name)
            if ap is not None:
                neg_mask = energy_negative_mask(conf, ap, pf)
        with h5py.File(feat_file, "r") as hf:
            strt_index_query = hf["start_index_query"][:][0]
            onset, offset = evaluate_prototypes(conf, hf, encoder, device, strt_index_query, neg_mask=neg_mask)
        name_arr.extend([audio_name] * len(onset))
        onset_arr.extend(onset); offset_arr.extend(offset)
        print(f"[eval] {audio_name}: {len(onset)} predicted events", flush=True)

    raw_csv = os.path.join(conf.path.work_dir, "Eval_out_raw.csv")
    pp_csv = os.path.join(conf.path.work_dir, "Eval_out_postproc.csv")
    pd.DataFrame({"Audiofilename": name_arr, "Starttime": onset_arr,
                  "Endtime": offset_arr}).to_csv(raw_csv, index=False)
    run_post_processing(conf.eval.get("post_proc", "fixed"), conf.path.eval_dir,
                        raw_csv, pp_csv, n_shots=conf.train.n_shot,
                        min_dur=conf.eval.get("min_dur", 0.200))

    overall = evaluate(pp_csv, conf.path.eval_dir, conf.experiment_name, "VAL", conf.path.work_dir)
    result = {
        "experiment_name": conf.experiment_name,
        "seed": conf.seed,
        "hyper_placement": list(conf.model.hyper_placement),
        "overall": {k: float(v) for k, v in overall.items()},
    }
    with open(os.path.join(conf.path.work_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[eval] {conf.experiment_name} seed={conf.seed} -> {overall}", flush=True)


def main():
    base = OmegaConf.load(os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))
    cli = OmegaConf.from_dotlist(sys.argv[1:])
    conf = OmegaConf.merge(base, cli)
    OmegaConf.resolve(conf)
    print(OmegaConf.to_yaml(conf))

    if conf.stage.features:
        os.makedirs(conf.path.feat_train, exist_ok=True)
        os.makedirs(conf.path.feat_eval, exist_ok=True)
        print("=== Feature extraction (train) ===", flush=True)
        extract_train(conf)
        print("=== Feature extraction (eval) ===", flush=True)
        extract_eval(conf)
    if conf.stage.train:
        if conf.train.get("contrast", False):
            run_train_contrastive(conf)
        else:
            run_train(conf)
    if conf.stage.eval:
        run_eval(conf)


if __name__ == "__main__":
    main()
