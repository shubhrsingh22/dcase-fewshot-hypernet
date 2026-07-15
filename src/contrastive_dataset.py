"""Dynamic hard-negative contrastive dataset (Surrey DCASE-2024 training recipe).

Ports ``PrototypeDynamicArrayDataSet`` + ``IdentityBatchSampler``: for every
positive class, positive segments are sampled from its annotated events and
"negative" segments from the **inter-event gaps within the same recordings**
(background between consecutive events). Each class yields a positive sample
(label ``2c``) and a hard-negative sample (label ``2c+1``). Features are PCEN,
extracted per recording on the fly and cached in memory, then z-score normalised
with the shared training statistics for consistency with evaluation.
"""

import os
from glob import glob
from itertools import chain

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from src.feature_extract import FeatureExtractor, extract_feature


class ContrastiveDynamicDataset(Dataset):
    def __init__(self, conf, mean, std):
        self.conf = conf
        self.mean, self.std = mean, std
        self.fe = FeatureExtractor(conf)
        self.fps = conf.features.sr / conf.features.hop_mel
        self.seg_len = int(round(conf.features.seg_len * self.fps))
        self.feat_cache = {}
        self.meta = {}
        self._build_meta()
        self.classes = list(self.meta.keys())
        self.class2int = {c: i for i, c in enumerate(self.classes)}
        # virtual length: many episodes worth of samples
        self.length = int(3600 * 8 / conf.features.seg_len)

    def __len__(self):
        return self.length

    def _all_csv(self):
        return [f for d, _, _ in os.walk(self.conf.path.train_dir)
                for f in glob(os.path.join(d, "*.csv"))]

    def _build_meta(self):
        for file in sorted(self._all_csv()):
            glob_cls = file.split("/")[-2]
            df = pd.read_csv(file, header=0, index_col=False)
            df_pos = df[(df == "POS").any(axis=1)]
            if len(df_pos) == 0:
                continue
            starts = [s - 0.025 for s in df_pos["Starttime"]]
            ends = [e + 0.025 for e in df_pos["Endtime"]]
            if "CALL" in df_pos.columns:
                cls_list = [glob_cls] * len(starts)
            else:
                cls_list = list(chain.from_iterable(
                    df_pos.columns[(df_pos == "POS").loc[idx]].values
                    for idx, _ in df_pos.iterrows()))
            audio = file[:-4] + ".wav"
            neg_start = 0.0
            for s, e, c in zip(starts, ends, cls_list):
                m = self.meta.setdefault(c, {"info": [], "neg_info": [], "file": []})
                m["info"].append((s, e))
                m["neg_info"].append((neg_start, s))  # gap before this event
                m["file"].append(audio)
                neg_start = e
            if audio not in self.feat_cache:
                self.feat_cache[audio] = extract_feature(audio, self.fe, self.conf)

    def _select(self, start, end, feat):
        s, e = int(start * self.fps), int(end * self.fps)
        s = max(0, s); e = min(e, feat.shape[0])
        total = e - s
        L = self.seg_len
        if total <= 0:
            x = np.zeros((L, feat.shape[1]), dtype=np.float32)
        elif total < L:
            x = np.tile(feat[s:e], (int(np.ceil(L / total)), 1))[:L]
        else:
            rs = int(np.random.uniform(s, e - L))
            x = feat[rs:rs + L]
        if x.shape[0] != L:
            x = np.pad(x, ((0, L - x.shape[0]), (0, 0)))
        return ((x - self.mean) / self.std).astype(np.float32)

    def _sample_pos(self, cls):
        m = self.meta[cls]
        i = np.random.randint(len(m["info"]))
        return self._select(*m["info"][i], self.feat_cache[m["file"][i]])

    def _sample_neg(self, cls):
        m = self.meta[cls]
        for _ in range(20):
            i = np.random.randint(len(m["neg_info"]))
            s, e = m["neg_info"][i]
            if e - s >= 0.2:
                return self._select(s, e, self.feat_cache[m["file"][i]])
        # fallback: use the last gap regardless of length
        return self._select(s, e, self.feat_cache[m["file"][i]])

    def __getitem__(self, idx):
        # idx is interpreted as a class index (yielded by IdentityBatchSampler).
        cls = self.classes[idx % len(self.classes)]
        pos = self._sample_pos(cls)
        neg = self._sample_neg(cls)
        c = self.class2int[cls]
        return pos, neg, c * 2, c * 2 + 1


class IdentityBatchSampler(Sampler):
    """Yields, per episode, k_way classes each repeated n_shot*2 times (as indices)."""

    def __init__(self, n_classes, n_way, samples_per_cls, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.samples_per_cls = samples_per_cls
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            classes = torch.randperm(self.n_classes)[:self.n_way]
            batch = torch.cat([c.repeat(self.samples_per_cls) for c in classes])
            yield batch.tolist()
