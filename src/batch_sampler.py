"""Episodic batch sampler (from the DCASE baseline / jakesnell prototypical-nets)."""

import numpy as np
import torch
import torch.utils.data as data


class EpisodicBatchSampler(data.Sampler):
    """Yields ``n_episodes`` batches, each with ``n_way`` classes x ``n_samples``."""

    def __init__(self, labels, n_episodes, n_way, n_samples):
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_samples = n_samples

        labels = np.array(labels)
        self.samples_indices = []
        for i in range(max(labels) + 1):
            ind = np.argwhere(labels == i).reshape(-1)
            self.samples_indices.append(torch.from_numpy(ind))

        if self.n_way > len(self.samples_indices):
            raise ValueError('"n_way" is higher than the number of classes')

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            batch = []
            classes = torch.randperm(len(self.samples_indices))[:self.n_way]
            for c in classes:
                l = self.samples_indices[c]
                pos = torch.randperm(len(l))[:self.n_samples]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
