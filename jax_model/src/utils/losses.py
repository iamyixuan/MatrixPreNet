import jax
import jax.numpy as jnp
from .DDOpt import DiracGamma


def condition_number_loss(model, U1, DD_mat):
    gammas = jax.vmap(model)(U1, DD_mat)
    Mopt = DiracGamma(U1, gammas)
    # need to construct the matrix of opt
    M_prime = construct_matrix(Mopt, U1.shape[0])
    M_inv = model.scale * M_prime + jnp.identity(M_prime.shape[-1])[None, ...]
    precond_sys = jnp.matmul(M_inv, DD_mat)
    cond_number = jnp.linalg.cond(precond_sys)
    return jnp.mean(cond_number)


def construct_matrix(opt, B, n=128):
    identity = jnp.identity(n)
    B_identity = jnp.repeat(identity[None, ...], B, axis=0)
    X = int(jnp.sqrt(n / 2))
    T = int(jnp.sqrt(n / 2))
    columns = []
    for i in range(B_identity.shape[1]):
        e_i = B_identity[:, :, i]
        e_i = e_i.reshape(B, X, T, 2)
        columns.append(opt.apply(e_i))
    M = jnp.stack(columns, axis=1).reshape(B, n, n)
    return M


if __name__ == "__main__":

    class Model:
        def __init__(self, scale):
            self.scale = scale

        def __call__(self, U1, DD_mat):
            return jnp.ones((3, 2, 2))

    dummy_model = Model(0.4)

    key = jax.random.PRNGKey(0)
    B = 10
    U = jax.random.normal(key, (B, 2, 8, 8), dtype=jnp.complex64)
    gammas = jax.random.normal(key, (B, 3, 2, 2), dtype=jnp.complex64)

    loss = condition_number_loss(
        dummy_model,
        U,
        jax.random.normal(key, (B, 128, 128), dtype=jnp.complex64),
    )
    print(loss)
