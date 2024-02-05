import jax
from typing import NamedTuple
from jax import lax, jit
import jax.numpy as jnp

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
            PCGState(x=x, r=r, p=p, gamma=gamma, iterations=state.iterations + 1),
            None,
        )

    # def cond(state):
    #     r = state.r
    #     gamma = state.gamma if M is _identity else jnp.vdot(r,r)
    #     return (gamma > max_gamma) & (state.iterations < max_iters)

    dummy_inputs = jnp.arange(max_iters)

    state, _ = lax.scan(body, init(), dummy_inputs)  # use lax.scan for differentiation
    return state


solve_from_jit = jit(solve_from, static_argnames=("A", "max_iters", "tol", "atol", "M"))


def solve(A, b, x0, max_iters=20, tol=1e-4, atol=0.0, M=_identity):
    # x0 = jnp.zeros_like(b)
    return solve_from_jit(A, b, x0, max_iters, tol, atol, M)
