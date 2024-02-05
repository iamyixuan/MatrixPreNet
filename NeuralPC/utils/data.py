import numpy as np
import glob
import pickle
import random
import os
import torch
import h5py
import jax
from torch.utils.data import Dataset, DataLoader
import scipy.sparse.linalg as spla
from .scaler import MinMaxScaler
from .utils import split_idx


class PreconditionerData(Dataset):
    def __init__(self, path, mode="train", flatten=False):
        super().__init__()

        data = np.load(path, allow_pickle=True)
        matrix = data["input_mat"][:, np.newaxis, :, :]
        M_ilu = data["M_ilu"][:, np.newaxis, :, :]
        M_amg = data["M_amg"][:, np.newaxis, :, :]

        if flatten:
            matrix = matrix.reshape(matrix.shape[0], -1)
            M_ilu = M_ilu.reshape(matrix.shape[0], -1)
            M_amg = M_amg.reshape(matrix.shape[0], -1)

        train_idx, val_idx, test_idx = split_idx(matrix.shape[0])
        if mode == "train":
            self.x = matrix[train_idx]
            self.y_ilu = M_ilu[train_idx]
            self.y_amg = M_amg[train_idx]
        elif mode == "val":
            self.x = matrix[val_idx]
            self.y_ilu = M_ilu[val_idx]
            self.y_amg = M_amg[val_idx]
        elif mode == "test":
            self.x = matrix[test_idx]
            self.y_ilu = M_ilu[test_idx]
            self.y_amg = M_amg[test_idx]
        else:
            raise Exception("Specify dataset!")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).float()
        y_ilu = torch.from_numpy(self.y_ilu[idx]).float()
        y_amg = torch.from_numpy(self.y_amg[idx]).float()
        return x, (y_ilu, y_amg)


def condition_number_approx(M_inv_A):
    u, s, v = torch.linalg.svd(M_inv_A)
    kappa = s[0] / s[-1]
    return kappa


class precodition_loss(torch.nn.Module):
    def __init__(self, alpha=1, beta=1, gamma=1):
        super(precodition_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, A, M):
        assert A.shape == M.shape
        A = A.reshape(A.shape[0], 100, 100)
        M = M.reshape(M.shape[0], 100, 100)
        batch_ls = []
        for i in range(A.shape[0]):
            M_inv = torch.inverse(M[i])
            M_inv_A = torch.mm(M_inv, A[i])
            try:
                L_kappa = condition_number_approx(M_inv_A)
            except:
                L_kappa = 0
            L_inv = 1 / (1 + torch.linalg.det(M[i]))
            L_approx = torch.mean(torch.pow(A[i] - M[i], 2))
            L = self.alpha * L_kappa + self.beta * L_inv + self.gamma * L_approx
            batch_ls.append(L)
        return torch.mean(torch.stack(batch_ls))


def split_idx(length, key):
    k = jax.random.PRNGKey(key)
    idx = jax.random.permutation(k, length)
    trainIdx = idx[: int(0.6 * length)]
    valIdx = idx[-int(0.4 * length) :]
    return trainIdx, valIdx


class Data(Dataset):
    def __init__(self, data, kappa) -> None:
        super().__init__()
        self.data = data
        self.kappa = kappa

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return np.array(self.data[index]), self.kappa


class ILUData(Dataset):
    def __init__(self, data, M, kappa) -> None:
        super().__init__()
        self.data = data
        self.M = M
        self.kappa = kappa

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return np.array(self.data[index]), np.array(self.M[index]), self.kappa


# Incomplete Chelosky dataset
class ICData(Dataset):
    def __init__(self, U1, z, r, kappa) -> None:
        super().__init__()
        self.U1 = U1
        self.z_real = z.real
        self.z_imag = z.imag
        self.r_real = r.real
        self.r_imag = r.imag
        self.kappa = kappa

    def __len__(self):
        return self.z_real.shape[0]

    def __getitem__(self, index):
        z = (self.z_real[index], self.z_imag[index])
        r = (self.r_real[index], self.r_imag[index])
        U = self.U1[index]
        return U, r, z, self.kappa

def create_dataLoader(data, batchSize, kappa, shuffle: bool, dataset="CG"):
    if dataset == "CG":
        dataset = Data(data, kappa)
    elif dataset == 'IC':
        dataset = ICData(U1=data[0], z=data[1], r=data[2], kappa=kappa)
    else:
        dataset = ILUData(data[0], data[1], kappa)
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle)
    return loader
