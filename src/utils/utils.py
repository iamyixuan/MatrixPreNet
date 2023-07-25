import numpy as np

def split_idx(idx_len):
    rs = np.random.RandomState(0)
    idx = rs.permutation(idx_len)
    train_size = int(.5 * idx_len)
    test_size = int(.25 * idx_len)

    train_idx = idx[:train_size]
    val_idx = idx[train_size:train_size + test_size]
    test_idx = idx[-test_size:]
    return train_idx, val_idx, test_idx