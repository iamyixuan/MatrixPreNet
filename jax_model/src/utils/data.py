import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.experimental import sparse
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


class U1DDatasetCOO(Dataset):
    def __init__(self, datapath, mode):

        data = torch.load(datapath)
        DD_mat = np.array(data["DD_mat"])[:200]
        adj_matrix = np.where(DD_mat[0] != 0.0, 1.0, 0.0)
        self.adj_matrix = sparse.BCOO.fromdense(adj_matrix)
        self.init_edge = np.ones((len(self.adj_matrix.indices), 1))
        U1 = np.array(data["U1"])[:200]
        # permute U1 axis to match (8, 8, 2)
        U1 = np.transpose(U1, (0, 2, 3, 1))

        train_idx, val_idx, test_idx = split_data(
            jnp.arange(len(U1)), train_ratio=0.8, random=True
        )

        if mode == "train":
            self.U1 = U1[train_idx]
            self.DD_mat = DD_mat[train_idx]
        elif mode == "val":
            self.U1 = U1[val_idx]
            self.DD_mat = DD_mat[val_idx]
        elif mode == "test":
            self.U1 = U1[test_idx]
            self.DD_mat = DD_mat[test_idx]

    def __len__(self):
        return len(self.U1)

    def __getitem__(self, idx):
        U1 = self.U1[idx]
        return U1, self.init_edge, self.DD_mat[idx]


class U1DDMaskDataset(U1DDataset):
    def __init__(self, datapath, mode):
        super().__init__(datapath, mode)

        self.mask = self.DD_mat[0] != 0.0

    def __getitem__(self, idx):
        U1, DD_mat = super().__getitem__(idx)
        return U1, DD_mat, self.mask


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = U1DDatasetCOO(
        "/Users/yixuan.sun/Documents/projects/Preconditioners/MatrixPreNet/data/U1_DD_matrices.pt",
        "train",
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for U1, E, DD in dataloader:
        print(U1.shape)
        print(E.shape)
        print(DD.shape)
        break
    print(dataset.adj_matrix.shape)
