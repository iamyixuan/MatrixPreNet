from scipy import misc
import jax.scipy as jsp
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random
import jax


# Create a noisy version by adding random Gaussian noise
key = random.PRNGKey(1701)
rand_input = random.normal(key, shape=(10, 2, 24, 24, 24, 24))
rand_kernel = random.normal(key, shape=(5, 2, 3, 3, 3, 3))

out = jax.lax.conv(rand_input, rand_kernel, (1, 1, 1, 1), "SAME")
print(out.shape)
