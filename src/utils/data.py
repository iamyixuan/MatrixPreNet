import numpy as np
import glob
import pickle
import random
import os
import torch
import h5py
from torch.utils.data import Dataset
from .scaler import MinMaxScaler
from .utils import split_idx


class PreconditionerData(Dataset):
    def __init__(self, path, mode='train'):
        super().__init__()

        data = np.load(path)
        matrix = data['input_matrices']
        matrix = matrix.reshape(matrix.shape[0], -1)
        preconditioner = data['output_vectors']

        train_idx, val_idx, test_idx = split_idx(matrix.shape[0])
        if mode=='train':
            self.x = matrix[train_idx]
            self.y = preconditioner[train_idx]
        elif mode=='val':
            self.x = matrix[val_idx]
            self.y = preconditioner[val_idx]
        elif mode=='test':
            self.x = matrix[test_idx]
            self.y = preconditioner[test_idx]
        else:
            raise Exception('Specify dataset!')
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).float()
        y = torch.from_numpy(self.y[idx]).float()
        return x, y
            