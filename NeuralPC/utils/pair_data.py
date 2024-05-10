import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ._base_data_loader import BaseDataLoader, BaseDataset

def get_dataset(data_name):
    if data_name == "pair":
        return PairDataset
    elif data_name == "u1":
        return U1Data
    else:
        raise ValueError(f"Data loader for {data_name} not found")


class PairDataset(BaseDataLoader):
    """data with input output pairs"""

    def __init__(self, data_dir, batch_size, shuffle, validation_split):
        super(PairDataset, self).__init__(
            data_dir, batch_size, shuffle, validation_split
        )
        self.init()

    def init(self):
        # load data here
        with open(self.data_dir, "rb") as f:
            data = pickle.load(f)
        self.x, self.y = (
            self.transform(data["x"]),
            self.transform(data["y"]).cfloat(),
        )

    def get_data_loader(self):
        x_train, y_train, x_val, y_val = self.split_validation()
        train_data = BaseDataset(x_train, y_train)
        val_data = BaseDataset(x_val, y_val)
        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=val_data.__len__(),
        )
        return train_loader, val_loader

    def split_validation(self):
        assert (
            self.validation_split > 0.0
        ), "Validation split must be greater than 0"

        data_size = self.x.shape[0]
        train_size = int(data_size * (1 - self.validation_split))
        idx = np.arange(data_size)
        rd = np.random.RandomState(42)
        idx = rd.permutation(idx)

        train_idx = idx[:train_size]
        val_idx = idx[train_size:]
        return (
            self.x[train_idx],
            self.y[train_idx],
            self.x[val_idx],
            self.y[val_idx],
        )

    def transform(self, x):
        if len(x.shape) != 2:
            x = x.reshape(x.shape[0], -1)
        return x


class U1Data(BaseDataset):
    def __init__(self, data_dir, batch_size, shuffle, validation_split):
        super(U1Data, self).__init__(
            data_dir, batch_size, shuffle, validation_split
        )
        self.init()

    def init(self):
        # load u1 configuration
        data = np.load(self.data_dir)
        # raise it to complex exponential
        data = np.exp(1j * data)
        self.x = torch.from_numpy(data).cdouble()

    def get_data_loader(self):
        x_train, x_val = self.split_validation()
        train_data = BaseDataset(x_train)
        val_data = BaseDataset(x_val)
        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=val_data.__len__(),
        )
        return train_loader, val_loader

    def split_validation(self):
        assert (
            self.validation_split > 0.0
        ), "Validation split must be greater than 0"

        data_size = self.x.shape[0]
        train_size = int(data_size * (1 - self.validation_split))
        idx = np.arange(data_size)
        rd = np.random.RandomState(42)
        idx = rd.permutation(idx)

        train_idx = idx[:train_size]
        val_idx = idx[train_size:]
        return (
            self.x[train_idx],
            self.x[val_idx],
        )
