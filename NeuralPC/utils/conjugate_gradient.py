import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import torch
from jax import jit, lax

"""
from https://towardsdatascience.com/implementing-linear-operators-in-python-with-google-jax-c56be3a966c2
"""


def _identity(x):
    return x


class PCGState(NamedTuple):
    x: jnp.ndarray
    r: jnp.ndarray
    p: jnp.ndarray
    gamma: float
    iterations: int


def solve_from(A, b, x0, max_iters=20, tol=1e-4, atol=0.0, M=_identity):
    # Boyd Conjugate Gradients slide 22
    b_norm_sqr = jnp.vdot(b, b)
    max_gamma = jnp.maximum(jnp.square(tol) * b_norm_sqr, jnp.square(atol))

    def init():
        r0 = b - A(x0)
        p0 = z0 = M(r0)
        gamma = jnp.vdot(r0, z0).astype(jnp.float32)
        return PCGState(x=x0, r=r0, p=p0, gamma=gamma, iterations=1)

    def body(state, _):  # add dummy var for lax.scan
        p = state.p
        Ap = A(p)
        alpha = state.gamma / jnp.vdot(p, Ap)
        x = state.x + alpha * p
        r = state.r - alpha * Ap
        z = M(r)
        gamma = jnp.vdot(r, z).astype(jnp.float32)
        beta = gamma / state.gamma
        p = z + beta * p
        # check if any of the variables are nan
        print("the residuals", jnp.min(jnp.abs(r)))
        return (
            PCGState(
                x=x, r=r, p=p, gamma=gamma, iterations=state.iterations + 1
            ),
            None,
        )

    # def cond(state):
    #     r = state.r
    #     gamma = state.gamma if M is _identity else jnp.vdot(r,r)
    #     return (gamma > max_gamma) & (state.iterations < max_iters)

    dummy_inputs = jnp.arange(max_iters)

    state, _ = lax.scan(
        body, init(), dummy_inputs
    )  # use lax.scan for differentiation
    return state


solve_from_jit = jit(
    solve_from, static_argnames=("A", "max_iters", "tol", "atol", "M")
)


def solve(A, b, x0, max_iters=20, tol=1e-4, atol=0.0, M=_identity):
    # x0 = jnp.zeros_like(b)
    return solve_from_jit(A, b, x0, max_iters, tol, atol, M)


# pytorch implementation

def cg_batch(
    A_bmm,
    B,
    M_bmm=None,
    X0=None,
    rtol=1e-8,
    atol=0.0,
    maxiter=None,
    verbose=False,
):
    """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.

    https://github.com/sbarratt/torch_cg/blob/master/torch_cg/cg_batch.py

    some modifications are made to the original code

    M_bmm: should be the mat-vect product using the preconditioner,
    in which we need to flatten the vector first and at the output reshape it back to the original shape.

    the vectors are of shape (B, X, T, 2), so need to pay attention to computing alpha and beta.

    also - the dot product should be for complex vectors!!!
    """

    residual_hist = []
    solution_hist = []
    K, x, t, m = B.shape

    if M_bmm is None:
        M_bmm = lambda x: x
    if X0 is None:
        X0 = M_bmm(B)
    if maxiter is None:
        maxiter = 5 * x * t * m

    # assert B.shape == (K, n, m)
    # assert X0.shape == (K, n, m)
    # assert rtol > 0 or atol > 0
    # assert isinstance(maxiter, int)

    X_k = X0
    R_k = B - A_bmm(X_k)
    Z_k = M_bmm(R_k)
    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    B_norm = torch.norm(B, dim=1)
    stopping_matrix = torch.max(rtol * B_norm, atol * torch.ones_like(B_norm))

    if verbose:
        print("%03s | %010s %06s" % ("it", "dist", "it/s"))

    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        start_iter = time.perf_counter()
        Z_k = M_bmm(R_k)

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2.conj() * Z_k2).sum(dim=(1, 2, 3))
            denominator[denominator == 0] = 1e-8
            beta = (R_k1.conj() * Z_k1).sum(dim=(1, 2, 3)) / denominator
            P_k = Z_k1 + beta[:, None, None, None] * P_k1

        denominator = (P_k.conj() * A_bmm(P_k)).sum(dim=(1, 2, 3))
        denominator[denominator == 0] = 1e-8
        alpha = (R_k1.conj() * Z_k1).sum(dim=(1, 2, 3)) / denominator
        X_k = X_k1 + alpha[:, None, None, None] * P_k
        R_k = R_k1 - alpha[:, None, None, None] * A_bmm(P_k)
        end_iter = time.perf_counter()

        residual = B - A_bmm(X_k)
        residual_norm = torch.norm(
            residual.reshape(residual.size(0), -1), dim=1
        ).mean()
        
        # record X_k
        solution_hist.append(X_k)

        if verbose:
            residual_hist.append(residual_norm)
            print(
                "%03d | %8.4e %4.2f"
                % (
                    k,
                    residual_norm,
                    1.0 / (end_iter - start_iter),
                )
            )

        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break



    end = time.perf_counter()

    if verbose:
        if optimal:
            print(
                "Terminated in %d steps (reached maxiter). Took %.3f ms."
                % (k, (end - start) * 1000)
            )
        else:
            print(
                "Terminated in %d steps (optimal). Took %.3f ms."
                % (k, (end - start) * 1000)
            )

    info = {"niter": k, "optimal": optimal}

    if verbose:
        info["residuals"] = residual_hist
        info["solutions"] = solution_hist
        info["B"] = B

    return X_k, info


class CG(torch.autograd.Function):

    def __init__(
        self,
        A_bmm,
        M_bmm=None,
        rtol=1e-3,
        atol=0.0,
        maxiter=None,
        verbose=False,
    ):
        self.A_bmm = A_bmm
        self.M_bmm = M_bmm
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.verbose = verbose

    def forward(self, B, X0=None):
        X, _ = cg_batch(
            self.A_bmm,
            B,
            M_bmm=self.M_bmm,
            X0=X0,
            rtol=self.rtol,
            atol=self.atol,
            maxiter=self.maxiter,
            verbose=self.verbose,
        )
        return X

    def backward(self, dX):
        dB, _ = cg_batch(
            self.A_bmm,
            dX,
            M_bmm=self.M_bmm,
            rtol=self.rtol,
            atol=self.atol,
            maxiter=self.maxiter,
            verbose=self.verbose,
        )
        return dB
