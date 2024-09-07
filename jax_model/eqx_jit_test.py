import equinox as eqx
import jax
import jax.numpy as jnp


# @eqx.filter_jit
def test(a, b, indices):
    o = a.at[indices].set(b.flatten())
    return o


a = jnp.zeros((20, 10, 10))
b = jnp.ones((20, 10))
mask = jnp.eye(10) == 1
mask = jnp.repeat(mask[None, :, :], 20, axis=0)
indices = jnp.nonzero(mask)
print(indices)
print(a[indices].shape)

a_out = test(a, b, indices)
print(a_out.shape)
print(a_out[3])
