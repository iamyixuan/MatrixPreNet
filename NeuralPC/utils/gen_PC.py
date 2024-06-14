import multiprocessing
from multiprocessing import Pool

import jax
import jax.numpy as jnp
import numpy as np
import pyamg
from jax.scipy.sparse.linalg import cg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, spilu, splu
from tqdm import tqdm

from NeuralPC.utils.dirac import DDOpt

jax.config.update("jax_enable_x64", True)


class GenPCVectorPairs:
    def __init__(self, U1, kappa=0.276, num_pairs=1) -> None:
        self.U1 = U1
        self.kappa = kappa
        self.num_pairs = num_pairs

    def genDataSingleU(self, n_samples, idx):
        U1 = self.U1[idx : idx + 1]  # isolate single U1 field
        A = self.getMatrix(U1)
        v = self._genRandomVector(n_samples, seed=0)

        M_lower, M_upper = self._decomp_LU(A)

        r_list = []
        z_list = []
        for i in range(n_samples):
            r, z = self._solveLU(M_lower, M_upper, v[i])
            r_list.append(r)
            z_list.append(z)
        return np.array(r_list), np.array(z_list)

    def getMatrix(self, U1):
        """
        Get the matrix of the original system
        """
        n = 128
        A = np.zeros((n, n)).astype(np.complex128)
        for i in range(n):
            A = self._getEntry(A, U1, i)
        return A

    def _decomp_LU(self, A):
        A_sparse = csr_matrix(A)
        lu = spilu(A_sparse, permc_spec="NATURAL")
        M_lower = lu.L
        M_upper = lu.U
        return M_lower, M_upper

    def _solveLU(self, M_lower, M_upper, v):
        M_lower_jax = jnp.array(M_lower.toarray())
        M_upper_jax = jnp.array(M_upper.toarray())
        z0 = jax.scipy.linalg.solve_triangular(M_lower_jax, v, lower=True)
        z1 = jax.scipy.linalg.solve_triangular(M_upper_jax, z0, lower=False)
        return v, z1

    def _getEntry(self, A, U1, i):
        n = A.shape[0]
        e_i = np.zeros(n).astype(np.complex128)
        e_i[i] = 1
        A[:, i] = DDOpt(e_i.reshape(1, 8, 8, 2), U1=U1, kappa=self.kappa).ravel()
        return A

    def _genRandomVector(self, n, seed):
        rs_real = np.random.RandomState(seed)
        rs_imag = np.random.RandomState(10000 - seed)
        real = rs_real.uniform(size=(n, 128))
        imag = rs_imag.uniform(size=(n, 128))
        return real + 1j * imag


def random_b(key, shape):
    # Generate random values for the real and imaginary parts
    real_part = 1 - jax.random.uniform(key, shape)
    imag_part = 1 - jax.random.uniform(jax.random.split(key)[1], shape)
    # Combine the real and imaginary parts
    complex_array = real_part + 1j * imag_part
    return complex_array


# obtain the matrix form of DDopt
def getMatrix(U1, kappa):
    n = 128  # 8 * 8 * 2
    A = np.zeros((n, n)).astype(np.complex128)
    for i in range(n):
        e_i = np.zeros(n).astype(np.complex128)
        e_i[i] = 1
        A[:, i] = DDOpt(e_i.reshape(1, 8, 8, 2), U1=U1, kappa=kappa).ravel()
    return A


def genDataPair(A, v):
    """
    This function calculates the Incomplete Chelosky preconditioner of A

    Then we can generate input output pairs (v, x) for
    supervised learning model traning.

    Args:
        A: the matrix representation of the original linear
        system.
        v: a random vector that M_inv act on.
    """
    A_sparse = csr_matrix(A)
    lu = splu(A_sparse, permc_spec="NATURAL")
    M_lower = lu.L
    M_upper = lu.U
    M_lower_jax = jnp.array(M_lower.toarray())
    M_upper_jax = jnp.array(M_upper.toarray())
    z0 = jax.scipy.linalg.solve_triangular(M_lower_jax, v, lower=True)
    z1 = jax.scipy.linalg.solve_triangular(M_upper_jax, z0, lower=False)
    return v, z1


def random_v(num_samples):
    real = np.random.uniform(size=(num_samples, 128))
    imag = np.random.uniform(size=(num_samples, 128))
    return real + 1j * imag


def genBatchData(U1, kappa, v_batch):
    """
    args:
        U1: U1 field to determine the original linear system.
        kappa: to determine the original linear system.
        v_batch: batch of random vectors for data generation.
    """

    U1 = np.exp(1j * U1).astype(np.complex128)  #

    data = {"v": [], "x": [], "U1": []}
    for i in range(U1.shape[0]):
        for j in tqdm(range(10)):
            A = getMatrix(U1=U1[i : i + 1], kappa=kappa)
            v, x = genDataPair(A, v_batch[i * 10 + j])
            data["v"].append(v)
            data["x"].append(x)
            data["U1"].append(U1[i])
    return data


def single_thread(args):
    U1, kappa, v_rand = args
    A = getMatrix(U1=U1, kappa=kappa)
    v, x = genDataPair(A, v_rand)
    return v, x, U1


def multiprocess_run(U1, kappa, v_batch):
    args_list = [
        (U1[i : i + 1], kappa, v_batch[i * 10 + j])
        for i in range(U1.shape[0])
        for j in range(10)
    ]

    with Pool(processes=12) as pool:
        results = pool.map(single_thread, args_list)

    # Unpack results
    data = {
        "v": [result[0] for result in results],
        "x": [result[1] for result in results],
        "U1": [result[2] for result in results],
    }
    return data


def solveWithOpt(A, b, steps, M=None):
    """
    args:
        U1: single instance of U1 field of shape (1, 2, 8, 8)
        kappa: a scalar.
        b: the RHS of Ax = b; random vector of shape (1, 8, 8, 2).
        steps: maximum steps a cg sovler takes.
    """

    def opt(v):
        return jnp.dot(A, v)

    if M is not None:
        if M == "AMG":
            print("Using AMG preconditioner...")

            def M(r):
                """
                In the CG solver we want to pass M^-1 such that
                z = M^-1r - r is the residual after each iteration.

                This mapping is equivalent to solving Mz=r, taking r as the input
                """
                ml = pyamg.ruge_stuben_solver(A)
                z = ml.solve(r)
                return z

        elif M == "IC":
            A_sparse = csr_matrix(A)
            lu = spilu(A_sparse, permc_spec="NATURAL")
            M_lower = lu.L
            M_upper = lu.U
            M_lower_jax = jnp.array(M_lower.toarray())
            M_upper_jax = jnp.array(M_upper.toarray())

            def M(r):
                z0 = jax.scipy.linalg.solve_triangular(M_lower_jax, r, lower=True)
                z1 = jax.scipy.linalg.solve_triangular(M_upper_jax, z0, lower=False)
                return z1

    # construct AMG preconditioner.

    x0 = jnp.zeros_like(b)

    x_sol, _ = cg(opt, b.ravel(), x0.ravel(), maxiter=steps, M=M)
    residual = b.ravel() - opt(x_sol)
    print(jnp.linalg.norm(residual))
    return


if __name__ == "__main__":
    U1 = np.load(
        "/Users/yixuan.sun/Documents/projects/Preconditioners/datasets/Dirac/precond_data/config.l8-N1600-b2.0-k0.276-unquenched.x.npy"
    )

    U1 = np.exp(1j * U1).astype(np.complex128)

    pc_gen = GenPCVectorPairs(U1=U1)
    r, z = pc_gen.genDataSingleU(n_samples=1000, idx=0)
    print(r.shape, z.shape)
    np.savez("../../../datasets/Dirac/U1_0-n1000_rz.npz", r=r, z=z)

    # cpu_count = multiprocessing.cpu_count()
    # print(cpu_count)

    # A = getMatrix(U1=U1[0:1], kappa=0.276)
    # print(A.shape)

    # key = jax.random.PRNGKey(0)
    # b = random_b(key, shape=(1, 8, 8, 2))
    # solveWithOpt(A=A, b=b, steps=20, M='IC')

    # # generate dataset
    # v_batch = random_v(U1.shape[0]*10)
    # #data = genBatchData(U1=U1, kappa=0.276, v_batch=v_batch)
    # data = multiprocess_run(U1=U1, kappa=0.276, v_batch=v_batch)
    # import pickle
    # with open( "/Users/yixuan.sun/Documents/projects/Preconditioners/datasets/Dirac/IC_pairs-10perU-config.l8-N1600-b2.0-k0.276.pkl", 'wb') as f:
    #     pickle.dump(data, f)


"""
# Assume A is your matrix for which you need a preconditioner

A = ...

# Create the AMG solver which acts as a preconditioner
ml = pyamg.ruge_stuben_solver(A)

# Define a function that applies the AMG preconditioner (M^{-1})
def apply_amg_preconditioner(v):
    return ml.solve(v, tol=1e-10)


from scipy.sparse.linalg import cg, LinearOperator
import numpy as np

# Define your right-hand side vector b
b = ...

# Create a linear operator for the preconditioner
M_inv = LinearOperator((A.shape[0], A.shape[1]), matvec=apply_amg_preconditioner)

# Solve Ax = b using CG with the AMG preconditioner
x, info = cg(A, b, M=M_inv)

# Check if the solution was successful
if info == 0:
    print("CG converged successfully.")
else:
    print("CG did not converge. Info:", info)


"""
