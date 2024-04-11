# test for CG solver
from functools import partial

import numpy as np
import torch

from NeuralPC.utils.conjugate_gradient import cg_batch
from NeuralPC.utils.dirac import DDOpt_torch

if __name__ == "__main__":
    U1 = np.load('../datasets/Dirac/precond_data/config.l8-N1600-b2.0-k0.276-unquenched.x.npy')
    U1 = np.exp(1j * U1)
    U1 = torch.from_numpy(U1).cdouble()

    lin_opt = partial(DDOpt_torch, U1=U1, kappa=0.276)
    b = torch.randn(U1.size(0), 8, 8, 2).cdouble()
    x0 = torch.zeros_like(b).cdouble()

    x, info = cg_batch(lin_opt, B=b, X0=x0, maxiter=100, verbose=True)
