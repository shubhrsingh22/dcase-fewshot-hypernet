"""Log-mel / PCEN feature extraction for DCASE few-shot bioacoustic detection.

Adapted from the DCASE ``deep_learning`` baseline (``Feature_extract.py``), the
system reproduced in thesis Chapter 3.3. Changes:

* log-mel (``power_to_db``) is the default feature (thesis Sec 3.3.5); PCEN is
  kept as an option to match the original baseline.
* modern-librosa keyword call, no 100-file cap, single bulk h5 write.

Training: every recording is sliced into ``seg_len``-second segments, each
inheriting the clip-level POS class label, and written to ``Mel_train.h5``.
Evaluation: for each validation recording, positive / negative / query segment
sets are written to ``<recording>.h5`` using the adaptive segment length of the
baseline (based on the 5 support shots).
"""

import os
from glob import glob
from itertools import chain

import h5py
import librosa
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


class FeatureExtractor:
    def __init__(self, conf):
        self.feature_type = conf.features.feature_type
        self.sr = conf.features.sr
        self.n_fft = conf.features.n_fft
        self.hop = conf.features.hop_mel
        self.n_mels = conf.features.n_mels
        self.fmax = conf.features.fmax
        self.fmin = conf.features.fmin

    def extract(self, audio):
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop,
            n_mels=self.n_mels, fmax=self.fmax, fmin=self.fmin,
        )
        if self.feature_type == "pcen":
            # Per-channel energy normalisation, default params (baseline). Audio is
            # already scaled by 2**32 in extract_feature, matching the baseline.
            feat = librosa.pcen(mel, sr=self.sr)
        else:  # log-mel (thesis)
            feat = librosa.power_to_db(mel, ref=np.max)
        return feat.astype(np.float32)


def extract_feature(audio_path, extractor, conf):
    y, _ = librosa.load(audio_path, sr=conf.features.sr)
    y = y * (2 ** 32)  # scaling per librosa PCEN recommendation (baseline)
    feat = extractor.extract(y)
    return feat.T  # (T, n_mels)


def _fix_len(patch, seg_len):
    """Force a patch to exactly ``seg_len`` frames (tile if short, truncate if long)."""
    seg_len = int(seg_len)
    n = patch.shape[0]
    if n == seg_len:
        return patch
    if n == 0:
        return None
    if n < seg_len:
        repeat = int(seg_len / n) + 1
        patch = np.tile(patch, (repeat, 1))
    return patch[:seg_len]


def time_2_frame(df, fps):
    # 25 ms margin around onset/offset (baseline).
    df.loc[:, "Starttime"] = df["Starttime"] - 0.025
    df.loc[:, "Endtime"] = df["Endtime"] + 0.025
    start = [int(np.floor(s * fps)) for s in df["Starttime"]]
    end = [int(np.floor(e * fps)) for e in df["Endtime"]]
    return start, end


def _segment_positive(feat, df_pos, glob_cls_name, seg_len, hop_seg, fps):
    """Slice POS events into fixed-length segments; return (patches, labels)."""
    patches, labels = [], []
    start_time, end_time = time_2_frame(df_pos, fps)

    if "CALL" in df_pos.columns:
        cls_list = [glob_cls_name] * len(start_time)
    else:
        cls_list = [
            df_pos.columns[(df_pos == "POS").loc[index]].values
            for index, _ in df_pos.iterrows()
        ]
        cls_list = list(chain.from_iterable(cls_list))

    assert len(start_time) == len(end_time) == len(cls_list)

    n_feat = feat.shape[0]
    for idx in range(len(start_time)):
        str_ind = max(0, min(start_time[idx], n_feat))
        end_ind = max(0, min(end_time[idx], n_feat))
        label = cls_list[idx]
        if end_ind - str_ind <= 0:
            continue
        if end_ind - str_ind > seg_len:
            shift = 0
            while end_ind - (str_ind + shift) > seg_len:
                p = _fix_len(feat[int(str_ind + shift):int(str_ind + shift + seg_len)], seg_len)
                if p is not None:
                    patches.append(p); labels.append(label)
                shift += hop_seg
            p = _fix_len(feat[end_ind - seg_len:end_ind], seg_len)
            if p is not None:
                patches.append(p); labels.append(label)
        else:
            p = _fix_len(feat[str_ind:end_ind], seg_len)
            if p is not None:
                patches.append(p); labels.append(label)
    return patches, labels


def extract_train(conf):
    extractor = FeatureExtractor(conf)
    fps = conf.features.sr / conf.features.hop_mel
    seg_len = int(round(conf.features.seg_len * fps))
    hop_seg = int(round(conf.features.hop_seg * fps))

    csv_files = [
        f for d, _, _ in os.walk(conf.path.train_dir)
        for f in glob(os.path.join(d, "*.csv"))
    ]
    all_patches, all_labels = [], []
    for f in sorted(csv_files):
        parts = f.split("/")
        glob_cls_name = parts[parts.index("Training_Set") + 1]
        df = pd.read_csv(f, header=0, index_col=False)
        audio_path = f[:-4] + ".wav"
        feat = extract_feature(audio_path, extractor, conf)
        df_pos = df[(df == "POS").any(axis=1)]
        patches, labels = _segment_positive(feat, df_pos, glob_cls_name, seg_len, hop_seg, fps)
        all_patches.extend(patches)
        all_labels.extend(labels)
        print(f"  {glob_cls_name}/{os.path.basename(audio_path)}: {len(patches)} segments")

    X = np.stack(all_patches).astype(np.float32)
    os.makedirs(conf.path.feat_train, exist_ok=True)
    hdf_tr = os.path.join(conf.path.feat_train, "Mel_train.h5")
    with h5py.File(hdf_tr, "w") as hf:
        hf.create_dataset("features", data=X)
        hf.create_dataset("labels", data=[s.encode() for s in all_labels], dtype="S20")
    print(f"[train] wrote {X.shape} to {hdf_tr}; {len(set(all_labels))} classes")
    return X.shape


def _adaptive_seg_len(max_len):
    if max_len < 100:
        return max_len
    elif max_len < 500:
        return max_len // 4
    return max_len // 8


def extract_eval(conf):
    extractor = FeatureExtractor(conf)
    fps = conf.features.sr / conf.features.hop_mel
    n_shot = conf.train.n_shot
    os.makedirs(conf.path.feat_eval, exist_ok=True)

    csv_files = [
        f for d, _, _ in os.walk(conf.path.eval_dir)
        for f in glob(os.path.join(d, "*.csv"))
    ]
    n_files = 0
    for f in sorted(csv_files):
        name = os.path.basename(f).split(".")[0]
        audio_path = f[:-4] + ".wav"
        df_eval = pd.read_csv(f, header=0, index_col=False)
        q = df_eval["Q"].to_numpy()
        start_time, end_time = time_2_frame(df_eval, fps)
        index_sup = np.where(q == "POS")[0][:n_shot]

        diffs = [end_time[i] - start_time[i] for i in index_sup]
        seg_len = _adaptive_seg_len(max(diffs))
        hop_seg = max(1, seg_len // 2)

        feat = extract_feature(audio_path, extractor, conf)
        strt_query = end_time[index_sup[-1]]
        end_neg = feat.shape[0] - 1

        n_feat = feat.shape[0]

        def add(lst, patch):
            p = _fix_len(patch, seg_len)
            if p is not None:
                lst.append(p)

        feat_pos, feat_neg, feat_query = [], [], []
        # Negative set: whole file sliced into segments.
        pos_ind = 0
        while end_neg - pos_ind > seg_len:
            add(feat_neg, feat[int(pos_ind):int(pos_ind + seg_len)])
            pos_ind += hop_seg
        add(feat_neg, feat[end_neg - seg_len:end_neg])
        # Positive set: the 5 support shots.
        for index in index_sup:
            si = max(0, min(int(start_time[index]), n_feat))
            ei = max(0, min(int(end_time[index]), n_feat))
            if ei - si <= 0:
                continue
            if ei - si > seg_len:
                shift = 0
                while ei - (si + shift) > seg_len:
                    add(feat_pos, feat[int(si + shift):int(si + shift + seg_len)])
                    shift += hop_seg
                add(feat_pos, feat[ei - seg_len:ei])
            else:
                add(feat_pos, feat[si:ei])
        # Query set: from end of 5th shot to end of file.
        q_ind = 0
        while end_neg - (strt_query + q_ind) > seg_len:
            add(feat_query, feat[int(strt_query + q_ind):int(strt_query + q_ind + seg_len)])
            q_ind += hop_seg
        add(feat_query, feat[end_neg - seg_len:end_neg])

        hdf_eval = os.path.join(conf.path.feat_eval, name + ".h5")
        with h5py.File(hdf_eval, "w") as hf:
            hf.create_dataset("feat_pos", data=np.stack(feat_pos).astype(np.float32))
            hf.create_dataset("feat_neg", data=np.stack(feat_neg).astype(np.float32))
            hf.create_dataset("feat_query", data=np.stack(feat_query).astype(np.float32))
            hf.create_dataset("start_index_query", data=np.array([strt_query]))
            hf.create_dataset("seg_len", data=np.array([seg_len]))
            hf.create_dataset("hop_seg", data=np.array([hop_seg]))
        n_files += 1
        print(f"  [eval] {name}: seg_len={seg_len} pos={len(feat_pos)} "
              f"neg={len(feat_neg)} query={len(feat_query)}")
    print(f"[eval] wrote {n_files} recording feature files to {conf.path.feat_eval}")
    return n_files
