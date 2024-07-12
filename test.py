# test for CG solver
from functools import partial

import numpy as np
import torch

from NeuralPC.model.models import PrecondCNN

if __name__ == "__main__":
    U1 = np.load('../datasets/Dirac/precond_data/config.l8-N1600-b2.0-k0.276-unquenched.x.npy')
    U1 = np.exp(1j * U1)
    U1 = torch.from_numpy(U1)[:32].cfloat()
    print('input shape', U1.shape)

    net = PrecondCNN(in_dim=2, out_dim=1, hidden_dim=32)
    #
    out = net(U1)
    print(out.shape)
