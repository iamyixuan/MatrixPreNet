import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def split_data(data_idx, train_ratio=0.8, random=True, key=None):
    if key is not None:
        key = jax.random.PRNGKey(key) 
    else:
        key = jax.random.PRNGKey(0)
    if random:
        data_idx = jax.random.permutation(key, data_idx)
    n = len(data_idx)
    n_train = int(n * train_ratio)
    train_idx = data_idx[:n_train]
    left_idx = data_idx[n_train:]

    val_idx = left_idx[: len(left_idx) // 2]
    test_idx = left_idx[len(left_idx) // 2 :]

    return train_idx, val_idx, test_idx


class U1DDataset(Dataset):
    def __init__(self, datapath, mode):

        data = torch.load(datapath)
        DD_mat = np.array(data["DD_mat"])
        U1 = np.array(data["U1"])
        # permute U1 axis to match (8, 8, 2)
        U1 = np.transpose(U1, (0, 2, 3, 1))

        train_idx, val_idx, test_idx = split_data(
            jnp.arange(len(U1)), train_ratio=0.8, random=True
        )

        if mode == "train":
            self.DD_mat = DD_mat[train_idx]
            self.U1 = U1[train_idx]
        elif mode == "val":
            self.DD_mat = DD_mat[val_idx]
            self.U1 = U1[val_idx]
        elif mode == "test":
            self.DD_mat = DD_mat[test_idx]
            self.U1 = U1[test_idx]

    def __len__(self):
        return len(self.U1)

    def __getitem__(self, idx):
        U1 = self.U1[idx]
        return U1, self.DD_mat[idx]
