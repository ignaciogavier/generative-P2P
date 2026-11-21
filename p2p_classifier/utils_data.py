"""Window dataset for the P2P detectors.

A window is a pair of files <prefix>_<k>_<hand>{acc,gyr}.npy (T' x 3 each), resampled to T = 150 samples (3 s at 50 Hz).
Labels come from the file name:
  synthetic  {p|n}{left|right}_<k>_{l|r}...      p = P2P movement of the simulated hand
  real       {p|n}left{p|n}right_<k>_{l|r}...    label of the wrist the window belongs to (l or r)
"""
import os

import numpy as np
from scipy.interpolate import CubicSpline

import torch
from torch.utils.data import Dataset

# 3 seconds window at fs = 50 Hz
T = 3 * 50

# Labels for synthetic dataset
LBL = {('nleft', 'l'): 0,  ('nleft', 'r'): 0,
       ('pleft', 'l'): 1,  ('pleft', 'r'): 1,
       ('nright', 'l'): 0, ('nright', 'r'): 0,
       ('pright', 'l'): 1, ('pright', 'r'): 1}

# Labels for real (healthy and stroke) datasets
LBL2 = {('nleftnright', 'l'): 0, ('nleftnright', 'r'): 0,
        ('pleftnright', 'l'): 1, ('pleftnright', 'r'): 0,
        ('nleftpright', 'l'): 0, ('nleftpright', 'r'): 1,
        ('pleftpright', 'l'): 1, ('pleftpright', 'r'): 1}

# Interpolation function for synth dataset with low sampling rate
interp = lambda x: CubicSpline(np.arange(len(x)), x, bc_type='natural')(np.linspace(0, 36, T, endpoint=False))


class CustomDataset(Dataset):
    def __init__(self, file_names, is_synth=True, is_stroke=False, augment=False, with_gyro=True, with_acc=True):
        self.file_names = file_names
        self.is_synth = is_synth
        self.is_stroke = is_stroke
        self.augment = augment
        self.with_gyro = with_gyro
        self.with_acc = with_acc

    def __len__(self):
        return len(self.file_names) // 2  # num of files divided by 2 (acc and gyr)

    def __getitem__(self, idx):
        files_idx = self.file_names[2*idx:2*idx+2]
        if len(files_idx) < 2: print(idx, self.file_names[0])  # Couldn't find acc/gyr for that index
        data = np.stack([np.load(f) for f in files_idx], axis=1)  # (T', 2, 3)
        if 'smooth2' in files_idx[0]: data = interp(data)

        data = data.reshape(data.shape[0], -1).T  # (6, T')
        if data.shape[1] < T: data = np.pad(data, ((0, 0), (0, data.shape[0] - T)))
        elif data.shape[1] > T: data = np.stack([np.interp(np.linspace(0, 3, T, False),
                                                           np.linspace(0, 3, data.shape[1], False),
                                                           d) for d in data], 0)  # (6, T)
        if not self.with_gyro: data = data[:3]  # (3, T)
        if not self.with_acc: data = data[3:]  # (3, T)

        if self.augment: data = self._augment_data(data)

        basename = os.path.basename(files_idx[0])
        if self.is_synth: label = LBL[(basename.split('_')[0], basename.split('_')[2][0])]
        else: label = LBL2[(basename.split('_')[0], basename.split('_')[2][0])]
        return {'data': torch.FloatTensor(data), 'label': torch.FloatTensor([label])}

    def _augment_data(self, data):
        if np.random.rand() > 0.5: data = self._jittering(data)
        if np.random.rand() > 0.5: data = self._scaling(data)
        if np.random.rand() > 0.5: data = self._time_warping(data)
        if np.random.rand() > 0.5: data = self._flipping(data)
        return data

    def _jittering(self, data, sigma=0.5):
        noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
        return data + noise

    def _scaling(self, data, sigma=0.3):
        factor = np.random.normal(loc=1.0, scale=sigma, size=(len(data), 1))
        return data * factor

    def _time_warping(self, data, sigma=0.2):
        scale = np.random.uniform(1 - sigma, 1)
        def warp(x, s=1): return np.interp(np.arange(len(x)) * s, np.arange(len(x)), x)
        return np.apply_along_axis(warp, 1, data, s=scale)

    def _flipping(self, data):
        return np.flip(data, axis=1).copy()
