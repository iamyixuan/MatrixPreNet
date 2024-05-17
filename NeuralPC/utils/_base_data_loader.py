from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import DataLoader, Dataset


class BaseDataLoader(ABC):
    def __init__(self, data_dir, batch_size, shuffle, validation_split):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation_split = validation_split

    @abstractmethod
    def init(self):
        pass

    def get_data_loader(self):
        if self.y is None:
            x_train, x_val = self.split_validation()
            train_data = BaseDataset(x_train)
            val_data = BaseDataset(x_val)
        else:
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

        if self.y is None:
            return (
                self.x[train_idx],
                self.x[val_idx],
            )
        else:
            return (
                self.x[train_idx],
                self.y[train_idx],
                self.x[val_idx],
                self.y[val_idx],
            )


class BaseDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]
