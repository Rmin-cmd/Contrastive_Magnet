import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import shutil
import pickle
import random


class EmotionDataset(Dataset):
    def __init__(self, data, graph, label, transform=None):
        self.data = data # n_samples * n_features
        self.graph = graph
        self.transform = transform
        self.label = torch.from_numpy(label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # The sample should not be across different videos
        # Calculate remaining samples considering floating-point timeStep
        # n_samples_remain_each = 30 - self.n_segs * self.timeStep
        #
        # # Calculate start and end indices for slicing
        # start_idx = int((idx * self.timeStep + n_samples_remain_each * np.floor(idx / self.n_segs)) * self.fs)
        # end_idx = int((idx * self.timeStep + self.timeLen + n_samples_remain_each * np.floor(idx / self.n_segs))
        #               * self.fs)

        # # Slice the data between the calculated indices
        # one_seq = self.data[:, start_idx:end_idx]
        data_real = self.data.reshape([-1, self.data.shape[2]])[idx]

        data_imag = self.data.reshape([-1, self.data.shape[2]])[idx]

        graph = self.graph[idx]

        # Retrieve the corresponding label
        one_label = self.label[idx]

        # Apply transformation if provided
        if self.transform:
            data_real = self.transform(data_real)
            data_imag = self.transform(data_imag)

        # Convert to PyTorch tensor and add an extra dimension for batch processing
        data_real = torch.FloatTensor(data_real)
        data_imag = torch.FloatTensor(data_imag)

        return graph, data_real, data_imag, one_label


class TrainSampler():
    def __init__(self, n_subs, batch_size, n_samples):
        self.n_per = int(np.sum(n_samples))
        self.n_subs = n_subs
        # Number of data points per session
        self.batch_size = batch_size
        self.n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples)))
        self.n_samples_per_trial = int(batch_size / len(n_samples))

        self.sub_pairs = []
        for i in range(self.n_subs):
            # j = i
            for j in range(i+1, self.n_subs):
                self.sub_pairs.append([i, j])
        random.shuffle(self.sub_pairs)
        # self.n_times = n_times

    def __len__(self):
        # return self.n_times * len(self.sub_pairs)
        return len(self.sub_pairs)

    def __iter__(self):
        for s in range(len(self.sub_pairs)):
            # for t in range(self.n_times):
            [sub1, sub2] = self.sub_pairs[s]

            ind_abs = np.zeros(0)
            if self.batch_size < len(self.n_samples_cum)-1:
                sel_vids = np.random.choice(np.arange(len(self.n_samples_cum)-1), self.batch_size)
                for i in sel_vids:
                    ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i+1]), 1, replace=False)
                    ind_abs = np.concatenate((ind_abs, ind_one))
            else:
                for i in range(len(self.n_samples_cum)-2):
                    # np.random.seed(i)
                    ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i+1]),
                                               self.n_samples_per_trial, replace=False)
                    ind_abs = np.concatenate((ind_abs, ind_one))

                i = len(self.n_samples_cum) - 2
                # np.random.seed(i)
                ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i + 1]),
                                           int(self.batch_size - len(ind_abs)), replace=False)
                ind_abs = np.concatenate((ind_abs, ind_one))
                # print('ind abs length', len(ind_abs))

            assert len(ind_abs) == self.batch_size

            ind_this1 = ind_abs + self.n_per*sub1
            ind_this2 = ind_abs + self.n_per*sub2

            batch = torch.LongTensor(np.concatenate((ind_this1, ind_this2)))
            # print(batch)
            yield batch