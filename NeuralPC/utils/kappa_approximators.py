import torch
import torch.nn as nn


class Ritz_estimator(nn.Module):
    """
    Ritz estimator class

    Args:
        lin_opt: linear operator
        lam: current E-value
        psi_all: current E-vector
        N_eig: number of E-vectors
        Rsd_r: target relative residual
        Rsd_a: target absolute residual
        zero_cutoff: if ev-slip is smaller than this, it is considered zero
        n_renorm: number of renormalizations
        n_min: minimum number of iterations
        n_max: maximum number of iterations
        MaxGC: maximum number of gradient calculations
        ProjApsiP: projection of A (?)
        n_count: no iters actually done (?)
        final_grad: final gradient norm
        Kalk_Sim: Are we in Kalk-Simma mode?
        delta_cycle: initial error estimate (KS mode)
        gamma_factor: convergence factor
    """

    def __init__(
        self,
        lin_opt,
        lam,
        psi_all,
        N_eig,
        Rsd_r,
        Rsd_a,
        zero_cutoff,
        n_renorm,
        n_min,
        n_max,
        MaxGC,
        ProjApsiP,
        n_count,
        final_grad,
        Kalk_Sim,
        delta_cycle,
        gamma_factor,
    ):
        super(Ritz_estimator, self).__init__()
        self.lin_opt = lin_opt
        self.lam = lam
        self.psi_all = psi_all
        self.N_eig = N_eig
        self.Rsd_r = Rsd_r
        self.Rsd_a = Rsd_a
        self.zero_cutoff = zero_cutoff
        self.n_renorm = n_renorm
        self.n_min = n_min
        self.n_max = n_max
        self.MaxGC = MaxGC
        self.ProjApsiP = ProjApsiP
        self.n_count = n_count
        self.final_grad = final_grad
        self.Kalk_Sim = Kalk_Sim
        self.delta_cycle = delta_cycle
        self.gamma_factor = gamma_factor

    def forward(self):
        N5 = self.psi_all.shape[0]
        pass


class Lanczos:
    """The Lanczos method reduces a large Hermitian matrix A into a smaller
    tridiagonal matrix T using an orthogonal basis constructed from the Krylov subspace.
    The eigenvalues of T are used as approximations to the eigenvalues of A"""

    def __init__(self, m):
        """A: function that does matrix-vector multiplication (batched)
        b: initial vector
        n_eig: number of eigenvalues to compute
        m: number of Lanczos vectors to use
        tol: tolerance for convergence"""

        self.m = m

    def batch_dot(self, v, w):
        # if v, w are complex, then we need to use the complex dot product
        return torch.einsum("bi, bi -> b", v.conj(), w)

    # def tri_diag(self, a, b, c, k1=1, k2=0, k3=1):
    #     """Constructs a  batch of tridiagonal matrices from the batched diagonals a, b, c"""
    #     print(a.shape, b.shape, c.shape)
    #     return (
    #         torch.diag_embed(a, k1)
    #         + torch.diag_embed(b, k2)
    #         + torch.diag_embed(c, k3)
    #     )
    #
    # def _lanczos(self, A, v):
    #     n = v.shape[-1]
    #     x, y = torch.zeros((v.shape[0], n)), torch.zeros((v.shape[0], n - 1))
    #     v2, beta = torch.zeros((v.shape[0], 1)), torch.zeros((v.shape[0], 1))
    #
    #     for i in range(n):
    #         w_prime = A(v)
    #         alpha = self.batch_dot(w_prime, v)
    #         w = w_prime - alpha * v - beta * v2
    #         print(w.shape)
    #         beta = torch.norm(w, dim=1)  # take norm along the vector dimension
    #         x[:, i] = alpha
    #
    #         if i < n - 1:
    #             y[:, i] = beta
    #
    #         v2 = v
    #         v = w / beta
    #     return self.tri_diag(y, x, y)

    def run(self, A, v):
        n = v.shape[-1]
        B = v.shape[0]
        m = self.m
        if m > n:
            m = n
        V = torch.zeros((B, m, n), dtype=v.dtype, device=v.device)
        T = torch.zeros((B, m, m), dtype=v.dtype, device=v.device)
        vo = torch.zeros((B, 1), dtype=v.dtype, device=v.device)
        beta = torch.zeros((B, 1), dtype=v.dtype, device=v.device)

        for j in range(self.m - 1):
            w = A(v)

            alfa = self.batch_dot(w, v)
            w = w - alfa.unsqueeze(1) * v - beta * vo

            beta = torch.sqrt(self.batch_dot(w, w).real)
            vo = v.clone()
            v = w / beta.unsqueeze(1)
            T[:, j, j] = alfa
            T[:, j, j + 1] = beta
            T[:, j + 1, j] = beta
            V[:, j, :] = v
            beta = beta.unsqueeze(1)
        w = A(v)
        alfa = self.batch_dot(w, v)
        w = w - alfa.unsqueeze(1) * v - beta * vo
        T[:, m - 1, m - 1] = self.batch_dot(w, v)
        V[:, m - 1] = w / torch.sqrt(self.batch_dot(w, v)).unsqueeze(1)
        return T


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n = 128

    rand_mat_real = torch.randn(n, n, dtype=torch.float32)
    rand_mat_imag = torch.randn(n, n, dtype=torch.float32)
    rand_mat = rand_mat_real + 1j * rand_mat_imag
    rand_mat = rand_mat @ rand_mat.conj().T

    def matrix_vector_prod(v):
        """v should have shape (B, n)"""
        # diag_elements = torch.tensor([10, 20, 6, 4.0, 5.0, 8], dtype=torch.float32)
        A = rand_mat.reshape(1, n, n)
        return torch.einsum("bij, bj -> bi", A, v)

    v = torch.randn(n, dtype=rand_mat.dtype)
    v /= torch.norm(v, dim=0)
    v = v.reshape(1, n)

    lanczos = Lanczos(m=10)
    T = lanczos.run(matrix_vector_prod, v)
    eigenvalues = torch.linalg.eigvals(
        T[0]
    ).abs()  # Get real part of eigenvalues
    real_eigenvalues = torch.linalg.eigvals(rand_mat).abs()

    # need to sort the absolute values of the eigenvalues

    sorted_estimates = eigenvalues.sort()[0]
    sorted_actual = real_eigenvalues.sort()[0]

    # print("Approximate eigenvalues:", sorted_estimates)
    # print("Actual eigenvalues:", sorted_actual)

    true_kappa = sorted_actual[-1] / sorted_actual[0]
    estimated_kappa = sorted_estimates[-1] / sorted_estimates[0]
    true_range = sorted_actual[-1] - sorted_actual[0]
    estimated_range = sorted_estimates[-1] - sorted_estimates[0]

    # print("Estimated kappa", estimated_kappa)
    # print("Actual kappa", sorted_actual[-1] / sorted_actual[0])
    #
    #
    # print("range of eigenvalues", sorted_estimates[-1] - sorted_estimates[0])
    # print("range of actual eigenvalues", sorted_actual[-1] - sorted_actual[0])

    fig, ax = plt.subplots()
    ax.set_box_aspect(1 / 3)
    ax.scatter(
        real_eigenvalues,
        0.2 * torch.ones_like(real_eigenvalues),
        marker="x",
        color="r",
        label="true",
    )
    ax.yaxis.set_visible(False)
    ax.scatter(
        sorted_estimates,
        0.3 * torch.ones_like(sorted_estimates),
        marker="x",
        color="b",
        label="lanczos",
    )
    ax.set_ylim(0.1, 0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2)
    plt.show()
    fig.savefig(
        f"../../figures/lanczos_trueK{true_kappa:.3f}_estK{estimated_kappa:.3f}_trueR{true_range:.3f}_estR{estimated_range:.3f}.png",
        bbox_inches="tight",
        dpi=300,
    )
