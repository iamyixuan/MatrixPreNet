from functools import partial

import numpy as np
import torch
import torch.nn as nn

from .conjugate_gradient import cg_batch
from .kappa_approximators import Lanczos


def getLoss(loss_name, **kwargs):
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
    elif loss_name == "SpectrumLoss":
        return SpectrumLoss
    elif loss_name == "CNNMatCondNumberLoss":
        return CNNMatCondNumberLoss()
    elif loss_name == "ConvComplexLoss":
        if "kind" in kwargs:
            kind = kwargs.get("kind", "LAL")
        return ConvComplexLoss(kind)
    elif loss_name == "MatConditionNumberLoss":
        if "mask" in kwargs:
            mask = torch.load("./data/DD_L_sparse_masks.pt")
            return MatConditionNumberLoss(mask=mask)
        else:
            raise ValueError("Mask not provided")

    elif loss_name == "KconditionLoss":
        if "mask" in kwargs:
            mask = torch.load("./data/DD_L_sparse_masks.pt")
            return KconditionLoss(mask=mask)
        else:
            raise ValueError("Mask not provided")
    else:
        raise ValueError(f"Loss {loss_name} not found")


class ConvComplexLoss(nn.Module):
    def __init__(self, kind="LLDD"):
        super().__init__()
        self.kind = kind

    def forward(self, DD, L):
        preconditioned = self.precond_opt(DD, L)
        condition_num = torch.linalg.cond(preconditioned)
        return condition_num.mean()

    def precond_opt(self, DD, L):
        L = L.squeeze()
        DD = DD.squeeze()
        L_T = L.conj().transpose(-2, -1)
        if self.kind == "LAL":
            return torch.bmm(L, torch.bmm(DD, L_T))
        elif self.kind == "LLDD":
            return torch.bmm(L, torch.bmm(L_T, DD))
        else:
            raise ValueError(f"kind {self.kind} not found")


class KconditionLoss(nn.Module):
    def __init__(self, mask=None):
        super().__init__()
        self.mask = mask
        if self.mask is not None:
            # broadcast the mask to the batch dimension
            self.maskDD = self.mask["DD"]
            self.maskL = self.mask["L"]

    def forward(self, DD_entries, L_entries):
        mat, LL, DD = self.precond_mat(DD_entries, L_entries)
        trace_precond = torch.diagonal(mat, dim1=-2, dim2=-1).sum(dim=-1)
        trace_DD = torch.diagonal(DD, dim1=-2, dim2=-1).sum(dim=-1)
        # check if LL diagonal is 0
        det_LL = torch.log(torch.diagonal(LL, dim1=-2, dim2=-1).real).sum(
            dim=-1
        )

        return torch.mean(trace_precond / trace_DD / det_LL).real

    def precond_mat(self, DD_entries, L_entries):
        # making L*DDL
        DD = self.reconstruct_mat(DD_entries, self.maskDD)
        L = self.reconstruct_mat(L_entries, self.maskL)
        # make sure the diagonal entries of L is greater than 1e-3
        diag_mask = torch.eye(L.size(-1), device=L.device, dtype=torch.bool)
        diag_elem = torch.maximum(
            L[:, diag_mask].real,
            torch.ones_like(L[:, diag_mask], dtype=torch.float) * 1e-3,
        )
        L[:, diag_mask] = diag_elem.to(L.dtype)

        preconditioned = torch.bmm(L.conj().transpose(-2, -1), DD)
        preconditioned = torch.bmm(preconditioned, L)
        LL = torch.bmm(L.conj().transpose(-2, -1), L)
        return preconditioned, LL, DD

    def reconstruct_mat(self, entries, mask):
        M = torch.zeros(
            (entries.size(0), mask.size(-1), mask.size(-1)),
            device=entries.device,
            dtype=entries.dtype,
        )
        M[:, mask] = entries
        return M


class MatConditionNumberLoss(nn.Module):
    def __init__(self, mask=None):
        super().__init__()
        self.mask = mask
        if self.mask is not None:
            # broadcast the mask to the batch dimension
            self.maskDD = self.mask["DD"]
            self.maskL = self.mask["L"]

    def forward(self, DD_entries, L_entries, scale):
        mat = self.precond_mat(DD_entries, L_entries, scale)
        # cond_num = torch.linalg.cond(mat)
        U, S, V = torch.linalg.svd(mat)
        cond_num = S[..., 0] / S[..., -1]
        return torch.mean(cond_num, dim=0)

    def precond_mat(self, DD_entries, L_entries, scale):
        # make LL_inv@DD matrix
        DD = self.reconstruct_mat(DD_entries, self.maskDD)
        epsilon = self.reconstruct_mat(L_entries, self.maskL) * scale
        identity = torch.eye(
            epsilon.size(-1), device=epsilon.device
        ).unsqueeze(0)
        identity = identity.repeat(epsilon.size(0), 1, 1)
        L = epsilon + identity

        LL = torch.bmm(L, L.conj().transpose(-2, -1))
        # LL_inv = torch.linalg.inv(LL)
        preconditioned = torch.bmm(LL, DD)
        return preconditioned

    def reconstruct_mat(self, entries, mask):
        assert mask.size(-1) == 128
        assert mask.size(-1) == mask.size(-2)
        M = torch.zeros(
            (entries.size(0), mask.size(-1), mask.size(-1)),
            device=entries.device,
            dtype=entries.dtype,
        )
        M[:, mask] = entries
        return M


class CNNMatCondNumberLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, DD_mat, L_epsilon, scale):
        mat = self.precond_mat(DD_mat, L_epsilon, scale)
        # cond_num = torch.linalg.cond(mat)
        U, S, V = torch.linalg.svd(mat)
        cond_num = S[..., 0] / S[..., -1]
        return torch.mean(cond_num, dim=0)

    def precond_mat(self, DD_mat, L_epsilon, scale):
        # make LL_inv@DD matrix
        identity = torch.eye(
            L_epsilon.size(-1), device=L_epsilon.device
        ).unsqueeze(0)
        identity = identity.repeat(L_epsilon.size(0), 1, 1)
        L = scale * L_epsilon + identity

        LL = torch.bmm(L, L.conj().transpose(-2, -1))
        # LL_inv = torch.linalg.inv(LL)
        preconditioned = torch.bmm(LL, DD_mat)
        return preconditioned


class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        raise NotImplementedError

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
        indices = torch.tril_indices(
            row=n, col=n, offset=0, device=net_out.device
        )
        matrices[:, indices[0], indices[1]] = net_out

        # Perform batched matrix-vector multiplication
        bmm = torch.bmm(
            matrices.conj().transpose(-2, -1), v.unsqueeze(-1)
        ).squeeze(-1)
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


class SpectrumLoss(BaseLoss):
    def __init__(self, DDOpt):
        super().__init__()
        self.DDOpt = DDOpt
        self.lanczos = Lanczos(m=10)

        self.n = 128
        self.matrix_getter = GetBatchMatrix(self.n)

    def forward(self, net_out, U1, use_true=False):
        def _matvect(x):
            # preconditioned system is LL*D*D
            x = x.reshape(x.size(0), 8, 8, 2)
            x = self.DDOpt(x, U1, kappa=0.276)
            # make sure it is LL*
            x = x.reshape(x.size(0), -1)
            x = self.upper_tri_mat(net_out, x)
            x = self.lower_tri_matvect(net_out, x)
            # output shape (B, 128)
            return x

        if use_true:
            mat = self.matrix_getter.getBatch(U1.shape[0], _matvect)
            condition_num = torch.linalg.cond(mat)
            return condition_num.mean()
        else:
            # random vector with norm 1
            v = torch.randn((net_out.shape[0], self.n), dtype=U1.dtype)
            v /= torch.norm(v, dim=1, keepdim=True)

            T = self.lanczos.run(_matvect, v)
            eigenvalues = torch.linalg.eigvals(T).abs()
            max_eigen = eigenvalues.max(dim=1)[
                0
            ]  # only return the values not the indices
            min_eigen = eigenvalues.min(dim=1)[0]
            # return torch.mean(max_eigen - min_eigen) + torch.mean(
            #     torch.relu(5 - min_eigen) ** 2
            # )
            return torch.mean(max_eigen**2) - torch.mean(min_eigen**2)

    def pc_spectrum(self, net_out, U1, use_true=False):
        """preconditioning system spectrum"""

        def _matvect(x):
            # preconditioned system is LL*D*D
            x = x.reshape(x.size(0), 8, 8, 2)
            x = self.DDOpt(x, U1, kappa=0.276)
            # make sure it is LL*
            x = x.reshape(x.size(0), -1)
            x = self.upper_tri_mat(net_out, x)
            x = self.lower_tri_matvect(net_out, x)
            # output shape (B, 128)
            return x

        if use_true:
            mat = self.matrix_getter.getBatch(U1.shape[0], _matvect)
            eigenvalues = torch.linalg.eigvals(mat).abs()
        else:
            # random vector with norm 1
            v = torch.randn((net_out.shape[0], self.n), dtype=U1.dtype)
            v /= torch.norm(v, dim=1, keepdim=True)

            T = self.lanczos.run(_matvect, v)
            eigenvalues = torch.linalg.eigvals(T).abs()
        return eigenvalues

    def org_spectrum(self, net_out, U1, use_true=False):
        """original system spectrum"""

        def _matvect(x):
            x = x.reshape(x.size(0), 8, 8, 2)
            x = self.DDOpt(x, U1, kappa=0.276)
            # make sure it is LL*
            x = x.reshape(x.size(0), -1)
            return x

        if use_true:
            mat = self.matrix_getter.getBatch(U1.shape[0], _matvect)
            eigenvalues = torch.linalg.eigvals(mat).abs()
        else:
            # random vector with norm 1
            v = torch.randn((net_out.shape[0], self.n), dtype=U1.dtype)
            v /= torch.norm(v, dim=1, keepdim=True)

            T = self.lanczos.run(_matvect, v)
            eigenvalues = torch.linalg.eigvals(T).abs()
        return eigenvalues

    def lower_spectrum(self, net_out, U1, use_true=False):
        """original system spectrum"""

        def _matvect(x):
            x = x.reshape(x.size(0), -1)
            x = self.lower_tri_matvect(net_out, x)
            # output shape (B, 128)
            return x

        if use_true:
            T = self.matrix_getter.getBatch(U1.shape[0], _matvect)
            eigenvalues = torch.linalg.eigvals(T).abs()
        else:
            # random vector with norm 1
            v = torch.randn((net_out.shape[0], self.n), dtype=U1.dtype)
            v /= torch.norm(v, dim=1, keepdim=True)

            T = self.lanczos.run(_matvect, v)
            eigenvalues = torch.linalg.eigvals(T).abs()
        return eigenvalues, T


class BasisOrthoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, basis_real, basis_imag):
        basis_real = basis_real.reshape(basis_real.size(0), -1)
        basis_imag = basis_imag.reshape(basis_imag.size(0), -1)
        basis = torch.complex(basis_real, basis_imag)
        dot_prod = torch.matmul(basis, basis.conj().T)
        return torch.norm(
            dot_prod - torch.eye(basis.size(0), device=basis.device)
        )


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


class K_Loss(BaseLoss):
    """K condition number loss"""

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

        # check if there is 0 in net_out
        if torch.any(net_out.real == 0):
            print("net_out contains 0")

        trace = self.trace_mat(U1, mat_vect)
        trace2 = self.trace_mat(U1, mat_vect, D=True)
        LL_det = self.det_L_mat(U1, L_mat_vect)
        """determinant values are across multiple orders of magnitude"""
        print(LL_det, "sdfsdfad")

        loss = trace / (trace2 * LL_det)
        print(loss, "asdfasdf")

        # check if there is nan in the loss
        if torch.any(torch.isnan(loss)):
            print("loss contains nan")
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
            log_diag_elem = torch.log(mat_vect(e)[:, i])
            log_diag += log_diag_elem

        return torch.abs(torch.exp(log_diag))


class GetBatchMatrix(nn.Module):
    def __init__(self, n) -> None:
        self.n = n

    def getBatch(self, B, mat_vec):
        """
        Get the matrix of the original system
        """
        A = torch.zeros((B, self.n, self.n)).cdouble()
        for i in range(self.n):
            A[:, :, i] = self._getColumn(A, i, mat_vec)
        return A

    def getMatrix(self, mat_vec):
        """
        Get the matrix of the original system
        """
        A = torch.zeros((self.n, self.n)).cdouble()
        for i in range(self.n):
            A = self._getColumn(A, i, mat_vec)
        return A

    def _getColumn(self, A, i, mat_vec):
        e_i = torch.zeros((A.shape[0], self.n)).cdouble()
        e_i[:, i] = 1
        # A[:, i] = mat_vec(e_i.reshape(1, 8, 8, 2)).ravel()
        B_col = mat_vec(e_i.reshape(A.shape[0], 8, 8, 2))
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
        lambda_min = self.get_smallest_eigen(
            U1, net_out, v, num_iter, lambda_max
        )

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
                self.DDOpt(basis[i : i + 1], U1, kappa=0.276).reshape(
                    U1.shape[0], -1
                )
            )  # each instance should be of shape B, 128

        # the formed matrix approximation should be of shape B, num_basis, 128
        x = torch.stack(output_matrix, dim=1)

        return x


# calculate the matrix form of the Dirac operator
# implemenent GetMatrix in pytorch


class ComplexMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.norm(x - y, dim=1).mean()


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


class CG_loss(BaseLoss):
    def __init__(self, DDOpt, kappa=0.276, maxiter=20, verbose=False):
        super().__init__()
        self.DDOpt = DDOpt
        self.kappa = kappa
        self.verbose = verbose
        self.maxiter = maxiter

    def forward(self, net_out, U1):
        def _matvect(x):
            # make sure it is LL*
            x = x.reshape(x.size(0), -1)
            x = self.upper_tri_mat(net_out, x)
            x = self.lower_tri_matvect(net_out, x)
            return x.reshape(x.size(0), 8, 8, 2)

        DDOpt = partial(self.DDOpt, U1=U1, kappa=self.kappa)

        b = torch.rand(net_out.size(0), 8, 8, 2).cdouble().to(net_out.device)
        x, info = cg_batch(
            DDOpt,
            b,
            M_bmm=_matvect,
            maxiter=self.maxiter,
            verbose=self.verbose,
        )
        residuals = DDOpt(x) - b
        residual_norm = torch.norm(
            residuals.reshape(residuals.size(0), -1), dim=1
        ).mean()
        if self.verbose:
            return residual_norm, info
        else:
            return residual_norm
