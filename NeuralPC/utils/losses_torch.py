from functools import partial

import numpy as np
import torch
import torch.nn as nn

from .conjugate_gradient import cg_batch


def getLoss(loss_name):
    if loss_name == "MSE":
        return nn.MSELoss
    elif loss_name == "MAE":
        return nn.L1Loss
    elif loss_name == "ComplexMSE":
        return ComplexMSE
    elif loss_name == "ComplexMAEMSE":
        return ComplexMAEMSE
    elif loss_name == "ConditionNumberLoss":
        return ConditionNumberLoss
    elif loss_name == "K_Loss":
        return K_Loss
    elif loss_name == "BasisOrthoLoss":
        return BasisOrthoLoss
    elif loss_name == "CG_loss":
        return CG_loss
    else:
        raise ValueError(f"Loss {loss_name} not found")


class BasisOrthoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, basis_real, basis_imag):
        basis_real = basis_real.reshape(basis_real.size(0), -1)
        basis_imag = basis_imag.reshape(basis_imag.size(0), -1)
        basis = torch.complex(basis_real, basis_imag)
        dot_prod = torch.matmul(basis, basis.conj().T)
        return torch.norm(dot_prod - torch.eye(basis.size(0), device=basis.device))


class MatrixConditionNumber(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, matrix):
        U, S, V = torch.linalg.svd(matrix)  # S contains the singular values

        max_s = S[0]
        min_s = S[-1]

        return max_s / min_s


class KconditionNum(nn.Module):
    def __init__() -> None:
        super().__init__()


class K_Loss(nn.Module):
    '''K condition number loss'''
    def __init__(self, DDOpt):
        super().__init__()
        self.DDOpt = DDOpt
        self.matrix_condition = MatrixConditionNumber()
        self.matrix_getter = GetBatchMatrix(128)

        # self.M_form = "lower_tri"

    def forward(self, net_out, U1):
        def mat_vect(x):
            # x = x.repeat(U1.shape[0], 1, 1, 1)
            x = self.upper_tri_mat(net_out, x.reshape(x.size(0), -1))
            x = x.reshape(x.size(0), 8, 8, 2)
            x = self.DDOpt(x, U1, kappa=0.276)
            x = self.lower_tri_matvect(net_out, x.reshape(x.size(0), -1))
            return x.view(x.size(0), -1)

        def L_mat_vect(x):
            x = self.lower_tri_matvect(net_out, x.reshape(x.size(0), -1))
            return x.view(x.size(0), -1)

        trace = self.trace_mat(U1, mat_vect)
        trace2 = self.trace_mat(U1, mat_vect, D=True)

        loss = trace / trace2  # (detrace2 + 1e-6)
        return torch.mean(loss)

    def trace_mat(self, U1, mat_vect, n=128, D=False):
        """
        mat_vect: matrix vector product
        n: size of the matrix
        """
        diag = 0
        for i in range(n):
            e = torch.zeros(U1.size(0), n).cdouble()
            e[:, i] = 1
            if D:
                elem = (
                    self.DDOpt(e.reshape(e.size(0), 8, 8, 2), U1, kappa=0.276)
                    .reshape(U1.size(0), -1)[:, i]
                    .real
                )
                diag = diag + elem
            else:
                diag += mat_vect(e)[:, i].real
        return diag

    def det_L_mat(self, U1, mat_vect, n=128):
        """
        mat_vect: matrix vector product
        n: size of the matrix
        """
        log_diag = 0
        for i in range(n):
            e = torch.zeros(U1.size(0), n).cdouble()
            e[:, i] = 1
            log_diag += torch.log(mat_vect(e)[:, i].real)
        return torch.exp(2*log_diag)

    def upper_tri_mat(self, net_out, v):
        """Linear map by the lower tranangular matrix
        The computation should support batch dimension.
        """
        num_entries = net_out.shape[1]
        n = int((torch.sqrt(torch.tensor(1.0) + 8 * num_entries) - 1) / 2)

        if v.shape[1] != n:
            raise ValueError(
                f"The vector size must match the matrix dimensions. Matrix size {n} but got vector size {v.size(1)}"
            )

        matrices = torch.zeros(v.size(0), n, n, device=net_out.device).cdouble()

        # Fill in the lower triangular part of each matrix
        indices = torch.tril_indices(row=n, col=n, offset=0, device=net_out.device)
        matrices[:, indices[0], indices[1]] = net_out

        # Perform batched matrix-vector multiplication
        bmm = torch.bmm(matrices.conj().transpose(-2, -1), v.unsqueeze(-1)).squeeze(-1)
        return bmm

    def lower_tri_matvect(self, net_out, v):
        """
        net_out: B, num_entries
        v: B, 128
        """
        dim = v.size(1)
        matvect = torch.zeros_like(v)

        start_id = 0
        for i in range(dim):
            row_len = i + 1
            matvect[:, i] = torch.einsum(
                "bi, bi -> b",
                net_out[:, start_id : start_id + row_len],
                v[:, 0:row_len],
            )
            start_id += row_len

        return matvect


class GetBatchMatrix(nn.Module):
    def __init__(self, n) -> None:
        self.n = n

    def getBatch(self, B, mat_vec):
        """
        Get the matrix of the original system
        """
        A = torch.zeros((B, self.n, self.n)).cfloat()
        for i in range(self.n):
            A[:, :, i] = self._getColumn(A, i, mat_vec)
        # for k in range(B):
        #     A = self.getMatrix(mat_vec)
        #     if k == 0:
        #         batch = A.reshape(1, self.n, self.n)
        #     else:
        #         batch = np.concatenate([batch, A.reshape(1, self.n, self.n)], axis=0)
        return A

    def getMatrix(self, mat_vec):
        """
        Get the matrix of the original system
        """
        A = torch.zeros((self.n, self.n)).cfloat()
        for i in range(self.n):
            A = self._getColumn(A, i, mat_vec)
        return A

    def _getColumn(self, A, i, mat_vec):
        e_i = torch.zeros(self.n).cfloat()
        e_i[i] = 1
        # A[:, i] = mat_vec(e_i.reshape(1, 8, 8, 2)).ravel()
        B_col = mat_vec(e_i.reshape(1, 8, 8, 2))
        return B_col.reshape(B_col.shape[0], -1)


class ConditionNumberLoss(nn.Module):
    def __init__(self, DDOpt):
        super().__init__()
        self.DDOpt = DDOpt

    def forward(self, net_out, U1):
        # calculate M_inv D D_dag v
        shape = (net_out.shape[0], 8, 8, 2)
        v_real = torch.rand(shape)
        v_imag = torch.rand(shape)

        # v of shape B, 8, 8, 2
        v = torch.complex(v_real, v_imag)
        condition_num = self.power_method(U1, net_out, v)
        return torch.norm(condition_num)

    def power_method(self, U1, net_out, v, num_iter=100):

        lambda_max = self.get_largest_eigen(U1, net_out, v, num_iter)
        lambda_min = self.get_smallest_eigen(U1, net_out, v, num_iter, lambda_max)

        max_abs = torch.abs(lambda_max)
        min_abs = torch.abs(lambda_min)
        return max_abs / min_abs

    def get_largest_eigen(self, U1, net_out, v, num_iter):
        for i in range(num_iter):
            v_k = self.matrix_vec_prod(U1, net_out, v)
            v = v_k / torch.norm(v_k)

        Av = self.matrix_vec_prod(U1, net_out, v)

        eigenvalue = self.b_dot(Av, v) / self.b_dot(v, v)
        return eigenvalue

    def get_smallest_eigen(self, U1, net_out, v, num_iter, max_eigen):
        max_eigen = max_eigen.view(-1, 1)

        for i in range(num_iter):
            v_k = self.spectral_shift(U1, net_out, v, max_eigen)
            v = v_k / torch.norm(v_k)

        Av = self.spectral_shift(U1, net_out, v, max_eigen)
        eigenvalue = self.b_dot(Av, v) / self.b_dot(v, v)

        return eigenvalue + max_eigen.view(
            -1,
        )

    def matrix_vec_prod(self, U1, net_out, v):
        v = v.reshape(v.shape[0], 8, 8, 2)
        v_temp = self.DDOpt(v, U1, kappa=0.276)
        return self.M(net_out, v_temp)

    def spectral_shift(self, U1, net_out, v, max_eigen):
        return self.matrix_vec_prod(U1, net_out, v) - max_eigen * v.reshape(
            v.shape[0], -1
        )

    def M(self, net_out, v):
        v = v.reshape(v.shape[0], -1)
        M_v = torch.einsum("bij, bj -> bi", net_out, v)
        return M_v

    def b_dot(self, v, w):
        return torch.einsum("bi, bi -> b", v, w.conj())


class DDApprox(nn.Module):
    def __init__(self, basis_size, DDOpt):
        # the last number of the hidden layer must be 128 * 128 for simple linear map
        super().__init__()

        self.DDOpt = DDOpt
        # Create basis of trainable vectors
        self.basis_real = nn.Parameter(torch.randn(basis_size, 8, 8, 2))
        self.basis_imag = nn.Parameter(torch.randn(basis_size, 8, 8, 2))

    def forward(self, U1):
        basis = torch.complex(self.basis_real, self.basis_imag)
        # x is a random vector that is used to create the
        output_matrix = []
        for i in range(basis.shape[0]):
            output_matrix.append(
                self.DDOpt(basis[i : i + 1], U1, kappa=0.276).reshape(U1.shape[0], -1)
            )  # each instance should be of shape B, 128

        # the formed matrix approximation should be of shape B, num_basis, 128
        x = torch.stack(output_matrix, dim=1)

        return x


# calculate the matrix form of the Dirac operator
# implemenent GetMatrix in pytorch


class GetMatrixDDOpt(nn.Module):
    def __init__(self, DDOpt, kappa=0.276) -> None:
        self.kappa = kappa
        self.DDOpt = DDOpt

    def getMatrix(self, U1):
        """
        Get the matrix of the original system
        """
        n = 128
        A = torch.zeros((n, n)).cfloat()
        for i in range(n):
            A = self._getEntry(A, U1, i)
        return A

    def _getEntry(self, A, U1, i):
        n = A.shape[0]
        e_i = torch.zeros(n).float()
        e_i[i] = 1
        A[:, i] = self.DDOpt(e_i.reshape(1, 8, 8, 2), U1=U1, kappa=self.kappa).ravel()
        return A


class ComplexMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.norm(x - y)


class ComplexMAEMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.norm(x - y, p=1) + torch.norm(x - y, p=2)


def getBatchMatrix(DDOpt, U1):
    getter = GetMatrixDDOpt(DDOpt)
    for i in range(U1.shape[0]):
        A = getter.getMatrix(U1[i : i + 1])
        if i == 0:
            batch = A.reshape(1, 128, 128)
        else:
            batch = np.concatenate([batch, A.reshape(1, 128, 128)], axis=0)
    return torch.from_numpy(batch).cfloat()


class CG_loss(nn.Module):
    def __init__(self, DDOpt, kappa=0.276, verbose=False):
        super().__init__()
        self.DDOpt = DDOpt
        self.kappa = kappa
        self.verbose = verbose

    def forward(self, net_out, U1):
        def _matvect(x):
            x = x.reshape(x.size(0), -1)
            x = self.lower_tri_matvect(net_out, x)
            x = self.upper_tri_mat(net_out, x)
            return x.reshape(x.size(0), 8, 8, 2)

        DDOpt = partial(self.DDOpt, U1=U1, kappa=self.kappa)

        b = torch.rand(net_out.size(0), 8, 8, 2).cdouble().to(net_out.device)
        x, info = cg_batch(DDOpt, b, M_bmm=_matvect, maxiter=20, verbose=self.verbose)
        residuals = DDOpt(x) - b
        residual_norm = torch.norm(
            residuals.reshape(residuals.size(0), -1), dim=1
        ).mean()
        if self.verbose:
            return residual_norm, info
        else:
            return residual_norm

    def upper_tri_mat(self, net_out, v):
        """Linear map by the lower triangular matrix
        The computation should support batch dimension.
        """
        num_entries = net_out.shape[1]
        n = int((torch.sqrt(torch.tensor(1.0) + 8 * num_entries) - 1) / 2)

        if v.shape[1] != n:
            raise ValueError(
                f"The vector size must match the matrix dimensions. Matrix size {n} but got vector size {v.size(1)}"
            )

        matrices = torch.zeros(
            v.size(0), n, n, device=net_out.device, dtype=net_out.dtype
        )

        # Fill in the lower triangular part of each matrix
        indices = torch.tril_indices(row=n, col=n, offset=0, device=net_out.device)
        matrices[:, indices[0], indices[1]] = net_out

        # Perform batched matrix-vector multiplication
        bmm = torch.bmm(matrices.conj().transpose(-2, -1), v.unsqueeze(-1)).squeeze(-1)
        return bmm

    def lower_tri_matvect(self, net_out, v):
        """
        net_out: B, num_entries
        v: B, 128
        """
        dim = v.size(1)
        matvect = torch.zeros_like(v)

        start_id = 0
        for i in range(dim):
            row_len = i + 1
            matvect[:, i] = torch.einsum(
                "bi, bi -> b",
                net_out[:, start_id : start_id + row_len],
                v[:, 0:row_len],
            )
            start_id += row_len

        return matvect
