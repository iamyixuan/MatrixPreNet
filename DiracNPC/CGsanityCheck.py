# check if the CG solver is used correctly
from jax.scipy.sparse.linalg import cg
from NeuralPC.utils.dirac import DDOpt
from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
from NeuralPC.utils.conjugate_gradient import solve
import flax.linen as nn

jax.config.update("jax_enable_x64", True)
steps=100

class MLP(nn.Module):                    # create a Flax Module dataclass

  @nn.compact
  def __call__(self, x):
    shape = x.shape
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(128)(x)                 # create inline Flax Module submodules
    x = nn.Dense(shape[1]*shape[2]*shape[3])(x)  
    x = x.reshape(shape)     # shape inference
    return x


model = MLP()                 # instantiate the MLP model

x = jnp.empty((4, 8,8, 2))            # generate random data
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
    x_sol = cg(A=operator, b=b, x0=x, M=precond)
    return x_sol

U1 = np.load('../../datasets/Dirac/precond_data/config.l8-N200-b2.0-k0.276-unquenched-test.x.npy')
U1 = jnp.exp(1j*U1).astype(jnp.complex128)

operator = partial(DDOpt, U1=U1,kappa=0.276)
y = operator(x=jnp.ones((200, 8, 8, 2)))

b = random_b(jax.random.PRNGKey(0), (200, 8, 8, 2)).astype(jnp.complex128)
print(jnp.vdot(b, b))
# M = random_b(jax.random.PRNGKey(1), (8, 8)).astype(jnp.complex128)

x_sol = runPCG(operator=operator, b=b, precond=None)
state = solve(operator, b, max_iters=1000)
print(state.x.shape)
residual = b - operator(state.x)

print(residual)
