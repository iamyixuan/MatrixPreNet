import numpy as np
import torch

from NeuralPC.model.neural_preconditioner import ConditionNumberLoss

# generate a random PD complex matrix
n = 128
A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
A = np.dot(A, A.T.conj())


def matrix_vec_prod(A, x):
    return torch.matmul(A, x).squeeze()


def spectral_shift(A, v, max_eigen):
    return matrix_vec_prod(A, v) - max_eigen * v


def get_largest_eigen(A, v, num_iter):
    for i in range(num_iter):
        v_k = matrix_vec_prod(A, v)
        v = v_k / torch.norm(v_k)

        Av = matrix_vec_prod(A, v)

        eigenvalue = torch.dot(Av, v) / torch.dot(v, v)
        return eigenvalue


def get_smallest_eigen(A, v, num_iter, max_eigen):

    for i in range(num_iter):
        v_k = spectral_shift(A, v, max_eigen)
        v = v_k / torch.norm(v_k)
    Av = spectral_shift(A, v, max_eigen)
    eigenvalue = torch.dot(Av, v) / torch.dot(v, v)

    return eigenvalue + max_eigen.view(
        -1,
    )

def compute_condition_number(A, num_iter=2000):
    n = A.shape[0]
    v = torch.rand(n).cfloat()
    v = v / torch.norm(v)

    max_eigen = get_largest_eigen(A, v, num_iter)
    print(max_eigen, "max_eigen")
    min_eigen = get_smallest_eigen(A, v, num_iter, max_eigen)
    print(min_eigen, "min_eigen")
    return torch.abs(max_eigen) / torch.abs(min_eigen)


# Calculate the condition number of A
cond_num = np.linalg.cond(A)
estimated_cond_num = compute_condition_number(torch.tensor(A, dtype=torch.complex64)).item()

print("cond_num:", cond_num)
print("estimated_cond_num:", estimated_cond_num)
