"""Data generators for training/validation (adapted from baseline Datagenerator.py).

* ``Datagen`` loads the training h5, oversamples to balance classes, splits into
  train/val (stratified), and z-score normalises with training statistics.
* ``DatagenTest`` loads a per-recording eval h5 (pos/neg/query) and applies the
  same normalisation.
"""

import os
import warnings

import h5py
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def class_to_int(label_array, class_set):
    label2indx = {label: index for index, label in enumerate(class_set)}
    return np.array([label2indx[label] for label in label_array])


def balance_class_distribution(X, Y):
    x_index = [[i] for i in range(len(X))]
    ros = RandomOverSampler(random_state=42)
    x_unifm, y_unifm = ros.fit_resample(x_index, Y)
    unifm_index = [i[0] for i in x_unifm]
    X_new = np.array([X[i] for i in unifm_index])
    Y_new = np.array([Y[i] for i in unifm_index])
    return X_new, Y_new


def norm_params(X):
    return np.mean(X), np.std(X)


class Datagen:
    def __init__(self, conf):
        hdf_path = os.path.join(conf.path.feat_train, "Mel_train.h5")
        with h5py.File(hdf_path, "r") as hf:
            self.x = hf["features"][:]
            self.labels = [s.decode() for s in hf["labels"][:]]

        class_set = sorted(set(self.labels))
        self.n_classes = len(class_set)
        self.y = class_to_int(self.labels, class_set)
        self.x, self.y = balance_class_distribution(self.x, self.y)
        array_train = np.arange(len(self.x))
        _, _, _, _, train_array, valid_array = train_test_split(
            self.x, self.y, array_train, random_state=42, stratify=self.y
        )
        self.train_index = train_array
        self.valid_index = valid_array
        self.mean, self.std = norm_params(self.x[train_array])

    def feature_scale(self, X):
        return (X - self.mean) / self.std

    def generate_train(self):
        train_array = sorted(self.train_index)
        valid_array = sorted(self.valid_index)
        X_train = self.feature_scale(self.x[train_array])
        Y_train = self.y[train_array]
        X_val = self.feature_scale(self.x[valid_array])
        Y_val = self.y[valid_array]
        return X_train, Y_train, X_val, Y_val


class DatagenTest(Datagen):
    def __init__(self, hf, conf):
        super().__init__(conf=conf)
        self.x_pos = hf["feat_pos"][:]
        self.x_neg = hf["feat_neg"][:]
        self.x_query = hf["feat_query"][:]
        self.hop_seg = hf["hop_seg"][:]

    def generate_eval(self):
        X_pos = self.feature_scale(self.x_pos)
        X_neg = self.feature_scale(self.x_neg)
        X_query = self.feature_scale(self.x_query)
        return X_pos, X_neg, X_query, self.hop_seg
