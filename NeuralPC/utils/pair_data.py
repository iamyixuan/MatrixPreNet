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
    elif data_name == "time_series":
        return TimeSeriesData
    elif data_name == "DD_matrix":
        return DDMatrixData
    elif data_name == "DD_IC":
        return DD_ICData
    elif data_name == "U1_DD":
        return U1_DD_data
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
        # train_loader = DataLoader(
        #     train_data,
        #     batch_size=self.batch_size,
        #     shuffle=self.shuffle,
        # )
        train_loader = DataLoader(
            train_data,
            batch_size=train_data.__len__(),
            shuffle=self.shuffle,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=val_data.__len__(),
        )
        return train_loader, val_loader

    def transform(self, x):
        if len(x.shape) != 2:
            x = x.reshape(x.shape[0], -1)
        return x


class DDMatrixData(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle,
        validation_split,
        double_precision=False,
    ):
        super(DDMatrixData, self).__init__(
            data_dir, batch_size, shuffle, validation_split
        )
        self.double_precision = double_precision
        self.init()

    def init(self):
        # load data here
        if self.double_precision:
            dd_matrix = torch.load(self.data_dir).cdouble()
        else:
            dd_matrix = torch.load(self.data_dir).cfloat()
        assert dd_matrix.dim() == 3, "DD matrix must be 3D"
        self.x = self.transform(dd_matrix)
        self.y = None

    def transform(self, x):
        if len(x.shape) != 4:
            # add channel dimension
            x = x.unsqueeze(1)
        return x


class DD_ICData(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle,
        validation_split,
        double_precision=False,
    ):
        super(DD_ICData, self).__init__(
            data_dir, batch_size, shuffle, validation_split
        )
        self.double_precision = double_precision
        self.init()

    def init(self):
        # load data here
        data = torch.load(self.data_dir)
        dd_matrix = data["A"]
        L_IC = data["L_IC"]
        self.x = self.transform(dd_matrix)
        self.y = self.transform(L_IC)

    def transform(self, x):
        x = torch.stack(x)
        mask = x[0].abs() != 0
        x = x[:, mask]
        if self.double_precision:
            return x.cdouble()
        else:
            return x.cfloat()


class U1_DD_data(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle,
        validation_split,
        double_precision=False,
    ):
        super(U1_DD_data, self).__init__(
            data_dir, batch_size, shuffle, validation_split
        )
        self.double_precision = double_precision
        self.init()

    def init(self):
        # load data here
        data = torch.load(self.data_dir)
        U1 = data["U1"]
        DD_mat = data["DD_mat"]
        self.x = self.transform(U1)
        self.y = self.transform(DD_mat)

    def transform(self, x):
        if self.double_precision:
            return x.cdouble()
        else:
            return x.cfloat()


class U1Data(BaseDataLoader):
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


class TimeSeriesData(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle,
        validation_split,
        hist_len=10,
        horizon=1,
        lstm=True,
    ):
        super().__init__(data_dir, batch_size, shuffle, validation_split)
        self.hist_len = hist_len
        self.horizon = horizon
        self.lstm = lstm

        self.init()

    def init(self):
        with open(self.data_dir, "rb") as f:
            data = pickle.load(f)
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)
        x = data[:, : self.hist_len, ...]
        y = data[:, self.hist_len + self.horizon, ...]

        if self.lstm:
            x = x.reshape(x.shape[0], x.shape[1], -1)
        else:
            x = x.reshape(x.shape[0], -1)

        y = y.reshape(y.shape[0], -1)

        # separate real and imag parts
        x_real = x.real
        x_imag = x.imag
        y_real = y.real
        y_imag = y.imag

        self.x = torch.cat([x_real, x_imag], dim=-1).float()
        self.y = torch.cat([y_real, y_imag], dim=-1).float()
