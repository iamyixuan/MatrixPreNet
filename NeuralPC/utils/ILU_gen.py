import numpy as np
from sklearn.datasets import make_sparse_spd_matrix
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, spilu
from tqdm import tqdm
import pyamg


class DiracEquation:
    def __init__(self) -> None:
        # Constants
        self.hbar = 1.0  # Planck's constant divided by 2pi (reduced)
        self.c = 1.0  # Speed of light
        self.m = 1.0  # Mass of particle
        self.N = 100  # Number of grid points
        self.dx = 0.1  # Grid spacing

    def generate_dirac_problem(self, N):
        # Gamma matrices for 1+1 dimensions
        gamma0 = np.array([[1, 0], [0, -1]], dtype=complex)
        gamma1 = np.array([[0, 1], [1, 0]], dtype=complex)

        # Construct Dirac operator matrix using finite differences
        main_diag = np.ones(N) * self.m * self.c**2
        off_diag = -0.5 * 1j * self.hbar * self.c / self.dx

        # Create Dirac operator for 1D (this is a block matrix)
        H = sp.block_diag([sp.diags(main_diag), sp.diags(-main_diag)])
        H1 = sp.diags(
            [off_diag, off_diag], [-1, 1], (N, N)
        )  # off-diagonal blocks for gamma1

        H += sp.block_diag([H1, -H1])
        b = np.random.rand(2 * N)  # Random right-hand side

        return H, b

    def get_Jacobi(self, A):
        diagonal = np.abs(A.diagonal())
        return diagonal

    def get_ILU(self, A, N):
        print(A.shape, "the input matrix shape is")
        ilu = spilu(A)
        Mx = lambda x: ilu.solve(x)
        M_ilu = LinearOperator((2 * N, 2 * N), Mx)
        M_ilu = linear_operator_to_matrix(M_ilu)
        return M_ilu

    def get_AMG(self, A):
        ml = pyamg.smoothed_aggregation_solver(A)
        M_amg = ml.aspreconditioner()
        M_amg = linear_operator_to_matrix(M_amg)
        return M_amg

    def gen_data(self, N_data, N):
        A, b = self.generate_dirac_problem(N)
        print(A)
        M_Jacobi = self.get_Jacobi(A)
        M_ilu = self.get_ILU(A, N)
        print(M_ilu.shape)
        # M_amg = self.get_AMG(A)

        return A, (M_Jacobi, M_ilu)


def linear_operator_to_matrix(linear_operator):
    """
    Convert a LinearOperator to a dense matrix.

    Parameters:
        - linear_operator: the input LinearOperator

    Returns:
        - ndarray: the dense matrix form of the linear operator
    """
    n = linear_operator.shape[1]
    I = np.eye(n)
    columns = [linear_operator.matvec(I[:, i]) for i in range(n)]
    return np.column_stack(columns)


def generate_data(N, seed):
    A = make_sparse_spd_matrix(N, random_state=seed)
    ilu = spilu(A)
    Mx = lambda x: ilu.solve(x)
    M_ilu = LinearOperator((N, N), Mx)
    M_ilu = linear_operator_to_matrix(M_ilu)
    # AMG PC
    ml = pyamg.smoothed_aggregation_solver(A)
    M_amg = ml.aspreconditioner()
    M_amg = linear_operator_to_matrix(M_amg)
    return A, M_ilu, M_amg


if __name__ == "__main__":
    N = 100
    n_samples = 2000

    input_mat = []
    M_ilu_PCs = []
    M_amg_PCs = []
    for i in tqdm(range(n_samples)):
        A, M_ilu, M_amg = generate_data(N=N, seed=10000 - i)
        assert isinstance(M_ilu, np.ndarray)
        assert isinstance(M_amg, np.ndarray)
        input_mat.append(A)
        M_ilu_PCs.append(M_ilu)
        M_amg_PCs.append(M_amg)
    input_mat = np.array(input_mat)
    M_ilu = np.array(M_ilu_PCs)
    M_amg = np.array(M_amg_PCs)
    # dr_eqn = DiracEquation()
    # input_mat, M = dr_eqn.gen_data(N_data=n_samples, N=N)

    #     # Convert the lists to NumPy arrays
    # input_mat = np.array(input_mat)
    # M_ilu = np.array(M[1])

    # Save the input matrices and output vectors to a NumPy file
    np.savez(
        f"../data/SPD-ILU-AMG-smoothed_aggregation.npz",
        input_mat=input_mat,
        M_ilu=M_ilu,
        M_amg=M_amg,
    )
