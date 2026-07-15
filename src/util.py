"""Prototypical loss and prototype-based evaluation (adapted from baseline util.py)."""

import os

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from src.datagen import DatagenTest


def euclidean_dist(x, y):
    n, m, d = x.size(0), y.size(0), x.size(1)
    if d != y.size(1):
        raise ValueError("embedding dim mismatch")
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support):
    """Episodic prototypical loss + accuracy (Snell et al. 2017)."""
    target_cpu = target.to("cpu")
    input_cpu = input.to("cpu")
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    support_idxs = list(map(supp_idxs, classes))
    prototypes = torch.stack([input_cpu[idx].mean(0) for idx in support_idxs])
    query_idxs = torch.stack(
        list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))
    ).view(-1)
    query_samples = input_cpu[query_idxs]

    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes).view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss, acc


def prototypical_loss_filter_negative(input, target, n_support):
    """Hard-negative contrastive prototypical loss (Surrey DCASE-2024 system).

    Even class ids are positive classes, odd ids are their in-recording negative
    distractors. Prototypes are built for *all* classes (pos + neg); queries are
    drawn only from positive classes and must be assigned to their own positive
    prototype against all prototypes (including the hard negatives).
    """
    target_cpu = target.to("cpu")
    input_cpu = input.to("cpu")
    classes = torch.unique(target_cpu)
    pos_classes = classes[classes % 2 == 0]
    n_classes = len(classes)
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    def query_idxs(c):
        return target_cpu.eq(c).nonzero()[n_support:]

    prototypes = torch.stack([input_cpu[supp_idxs(c)].mean(0) for c in classes])
    q_idx = torch.stack([query_idxs(c) for c in pos_classes]).view(-1)
    query_samples = input_cpu[q_idx]

    dists = euclidean_dist(query_samples, prototypes)
    n_pos = len(pos_classes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_pos, n_query, -1)
    # positive prototypes sit at even indices 0,2,4,... in `prototypes`.
    target_inds = (torch.arange(0, n_pos) * 2).view(n_pos, 1, 1).expand(n_pos, n_query, 1).long()
    loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss, acc


def get_probability(pos_proto, neg_proto, query_out):
    """Softmax probability of each query segment belonging to the positive class."""
    prototypes = torch.stack([pos_proto, neg_proto]).squeeze(1)
    dists = euclidean_dist(query_out, prototypes)
    prob = torch.softmax(-dists, dim=1)
    return prob[:, 0].detach().cpu().tolist()


def probs_to_events(prob_final, thresh, hop_seg, strt_index_query, conf):
    """Turn an averaged per-segment probability array into onset/offset times."""
    hop_seg = int(hop_seg[0]) if hasattr(hop_seg, "__len__") else int(hop_seg)
    prob_thresh = np.where(prob_final > thresh, 1, 0)
    changes = np.convolve(np.array([1, -1]), prob_thresh)
    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]
    fps_factor = conf.features.hop_mel / conf.features.sr
    str_time_query = strt_index_query * fps_factor
    onset = onset_frames * hop_seg * fps_factor + str_time_query
    offset = offset_frames * hop_seg * fps_factor + str_time_query
    return onset, offset


def energy_negative_mask(conf, audio_path, pos_frames):
    """Energy(RMS)-based negative-region mask (Surrey `negative_onset_offset_estimate`).

    Marks frames whose RMS is below the midpoint of the positive-shot min/mean
    (quiet background) or above the positive max (loud outliers) as negative.
    """
    import librosa
    x, _ = librosa.load(audio_path, sr=conf.features.sr)
    rms = librosa.feature.rms(y=x, frame_length=conf.features.n_fft,
                              hop_length=conf.features.hop_mel)[0]
    pos_vals = [rms[s:e] for s, e in (pos_frames or []) if e > s and s < len(rms)]
    pos_vals = np.concatenate(pos_vals) if pos_vals else rms
    mn, mx, mean = float(np.min(pos_vals)), float(np.max(pos_vals)), float(np.mean(pos_vals))
    mask = (rms < (mn + mean) / 2) | (rms > mx)
    if mask.sum() < 435:
        mask = rms < (np.max(rms) / 6)
    return mask


def eval_file_meta(conf, name):
    """Resolve the validation audio path and first-``n_shot`` POS event frames."""
    import glob
    import pandas as pd
    matches = glob.glob(os.path.join(conf.path.eval_dir, "*", name + ".wav"))
    if not matches:
        return None, None
    audio_path = matches[0]
    df = pd.read_csv(audio_path[:-4] + ".csv")
    fps = conf.features.sr / conf.features.hop_mel
    idx = np.where(df["Q"].to_numpy() == "POS")[0][:conf.train.n_shot]
    frames = []
    for i in idx:
        s = int(np.floor((df["Starttime"].iloc[i] - 0.025) * fps))
        e = int(np.floor((df["Endtime"].iloc[i] + 0.025) * fps))
        frames.append((max(0, s), max(0, e)))
    return audio_path, frames


def compute_probs(conf, hdf_eval, encoder, device, neg_mask=None):
    """Encoder-side of evaluation: averaged positive-class probability per query segment.

    If ``neg_mask`` (a per-frame background mask) is given, the negative prototype
    is drawn only from segments that fall in energy-detected background regions
    rather than uniformly across the recording.
    """
    gen_eval = DatagenTest(hdf_eval, conf)
    X_pos, X_neg, X_query, hop_seg = gen_eval.generate_eval()
    X_pos = torch.tensor(X_pos)
    X_neg = torch.tensor(X_neg)
    if neg_mask is not None and len(X_neg) > 0:
        hs = int(hop_seg[0]) if hasattr(hop_seg, "__len__") else int(hop_seg)
        L = int(hdf_eval["seg_len"][:][0])
        keep = [i for i in range(len(X_neg))
                if neg_mask[i * hs: i * hs + L].mean() > 0.5]
        if len(keep) >= 10:
            X_neg = X_neg[keep]
    X_query = torch.tensor(X_query)
    Y_query = torch.LongTensor(np.zeros(X_query.shape[0]))
    q_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_query, Y_query),
        batch_size=conf.eval.query_batch_size, shuffle=False,
    )
    encoder.eval()
    with torch.no_grad():
        pos_proto = encoder(X_pos.to(device)).mean(dim=0)
        prob_comb = []
        for _ in range(conf.eval.iterations):
            neg_idx = torch.randperm(len(X_neg))[:conf.eval.samples_neg]
            neg_proto = encoder(X_neg[neg_idx].to(device)).mean(dim=0)
            prob_iter = []
            for x_q, _ in q_loader:
                x_query = encoder(x_q.to(device))
                prob_iter.extend(get_probability(pos_proto, neg_proto, x_query))
            prob_comb.append(prob_iter)
    prob_final = np.mean(np.array(prob_comb), axis=0)
    return prob_final, hop_seg


def evaluate_prototypes(conf, hdf_eval, encoder, device, strt_index_query, neg_mask=None):
    """Detect target events in one validation recording; return onset/offset arrays."""
    prob_final, hop_seg = compute_probs(conf, hdf_eval, encoder, device, neg_mask=neg_mask)
    onset, offset = probs_to_events(prob_final, conf.eval.threshold, hop_seg,
                                    strt_index_query, conf)
    assert len(onset) == len(offset)
    return onset, offset
