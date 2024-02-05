# check if the CG solver is used correctly
from jax.scipy.sparse.linalg import cg
from NeuralPC.utils.dirac import DDOpt
from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
from NeuralPC.utils.conjugate_gradient import solve
import flax.linen as nn
from NeuralPC.model.linearOpt import linearConvOpt, linearOpt

jax.config.update("jax_enable_x64", True)
steps = 100


class MLP(nn.Module):  # create a Flax Module dataclass
    @nn.compact
    def __call__(self, x):
        shape = x.shape
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)  # create inline Flax Module submodules
        x = nn.Dense(shape[1] * shape[2] * shape[3])(x)
        x = x.reshape(shape)  # shape inference
        return x


model = MLP()  # instantiate the MLP model

x = jnp.empty((4, 8, 8, 2))  # generate random data
params = model.init(jax.random.key(42), x)


def NNopt(x):
    return model.apply(params, x)


def random_b(key, shape):
    # Generate random values for the real and imaginary parts
    real_part = 1 - jax.random.uniform(key, shape)
    imag_part = 1 - jax.random.uniform(jax.random.split(key)[1], shape)
    # Combine the real and imaginary parts
    complex_array = real_part + 1j * imag_part
    return complex_array


def runPCG(operator, b, precond=None):
    x = jnp.zeros(b.shape).astype(b.dtype)
    x_sol = solve(A=operator, b=b, x0=x, M=precond)
    return x_sol


# def solveWithOpt(U1, b, steps):
#     def opt(x):
#         return DDOpt(x, U1, kappa=0.276)
#     x0 = jnp.zeros_like(b)
#     #x_sol, _ = cg(opt, b, x0, maxiter=steps, tol=1e-8)
#     state = solve(opt, b, steps)
#     #return x_sol
#     return state.x


def solveWithOpt(U1, b, kernels, steps):
    def linOpt(x):
        # x = x[:, jnp.newaxis, ...] # dummy channel axis
        # new_vect = linearOpt(x, w, bias)
        new_vect = linearConvOpt(x, kernels)
        return new_vect

    def opt(x):
        return DDOpt(x, U1, kappa=0.276)

    x0 = jnp.zeros_like(b)
    print(linOpt(x0).shape)
    # x_sol, _ = cg(opt, b, x0, maxiter=steps, tol=1e-8)
    state, _ = cg(opt, b, x0, maxiter=steps, M=linOpt)
    # return x_sol
    return state


U1 = np.load(
    "../../datasets/Dirac/precond_data/config.l8-N200-b2.0-k0.276-unquenched-test.x.npy"
)
U1 = jnp.exp(1j * U1).astype(jnp.complex128)[:, jnp.newaxis, ...]  # (B, 2, X, T)

operator = partial(DDOpt, U1=U1[:, 0, ...], kappa=0.276)
# y = operator(x=jnp.ones(( 1, 8, 8, 2)))

b = random_b(jax.random.PRNGKey(0), (200, 1, 8, 8, 2)).astype(jnp.complex128)
# M = random_b(jax.random.PRNGKey(1), (8, 8)).astype(jnp.complex128)
w = jnp.tile(jnp.eye(128, 128), (200, 1, 1))
bias = jnp.zeros(shape=(200, 128))

kernels = jnp.ones(shape=(200, 8, 8, 4, 4))

vmap_solve = jax.vmap(solveWithOpt, in_axes=[0, 0, 0, None], out_axes=0)
res_l2norm = []
for s in range(20, 500, 10):
    # x_sol = runPCG(operator=operator, b=b, precond=None)
    state = vmap_solve(U1, b, kernels, s)
    state = state[:, 0, ...]
    # print(state.shape)
    # state  = solveWithOpt(U1[0], b[0], s)
    print(state.shape)

    # state = solve(operator, b[0], s)

    residual = b[:, 0, ...] - operator(state)

    residual = residual.reshape(residual.shape[0], -1)
    print(jnp.linalg.norm(residual[0]))
    norm = jnp.mean(jnp.linalg.norm(residual, axis=1))
    res_l2norm.append(norm)
    print(s, res_l2norm[-1])

# jnp.save('residual1.npy', jnp.array(res_l2norm))
