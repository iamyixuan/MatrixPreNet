import numpy as np
from functools import partial
import torch
from scipy.linalg import cholesky
from tqdm import tqdm

from NeuralPC.utils.conjugate_gradient import cg_batch
from NeuralPC.utils.dirac import DDOpt_torch as DDOpt
from NeuralPC.utils.losses_torch import GetBatchMatrix



# choleksy decomposition
def cholesky(A):
    # convert to numpy
    mask = A.abs() == 0
    L = torch.cholesky(A)
    L[mask] = 0
    return L

if __name__ == "__main__":
    # generate data
    dataPath = "/Users/yixuan.sun/Documents/projects/Preconditioners/datasets/Dirac/precond_data/config.l8-N1600-b2.0-k0.276-unquenched.x.npy"
    U1 = np.load(dataPath)
    U1 = torch.from_numpy(U1).to(torch.cdouble)
    U1 = torch.exp(1j * U1)
    print(U1.shape)

    def gen_x(size):
        real = torch.randn(size, 8, 8, 2)
        imag = torch.randn(size, 8, 8, 2)
        return real + 1j * imag

    # x = []
    # y = []
    #
    # for j in tqdm(range(128 * 10)):
    #     inputs = gen_x(U1.shape[0])
    #     x.append(inputs)
    #     y.append(DDOpt(inputs, U1, 0.276))
    #
    # x = torch.cat(x, dim=0)
    # y = torch.cat(y, dim=0)
    # print(x.shape, y.shape)
    #
    # data = {"x": x, "y": y}
    #
    # import pickle
    #
    # with open("./data/linear_inv_data_singleU1.pkl", "wb") as f:
    #     pickle.dump(data, f)

    # run CG solve on the system and save the data every stepk
    # for i inrange(10):gt
    opt = partial(DDOpt, U1=U1, kappa=0.276)
    matrix_getter = GetBatchMatrix(128)
    dd_matrices = matrix_getter.getBatch(U1.shape[0], opt)
    torch.save(dd_matrices, "./data/DD_matrices.pt")






    # B = gen_x(U1.shape[0])
    # x, info = cg_batch(
    #     opt,
    #     B,
    #     verbose=True,
    # )
    # solutions = torch.stack(info["solutions"], dim=1)
    # import pickle
    # with open("./data/DD_CG_solutions.pkl", "wb") as f:
    #     pickle.dump(solutions, f)
    #
    # print(solutions.shape)
