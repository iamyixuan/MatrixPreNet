import jax
import jax.numpy as jnp
from jax.experimental import sparse

from .DDOpt import Dirac_Matrix, DiracGamma


def HPD_opt(Dirac_Matrix_inst, x):
    Dx = Dirac_Matrix_inst.apply(Dirac_Matrix_inst.apply(x), dagger=True)
    return Dx


def inverse_loss(model, inputs):
    U1 = inputs[0]
    key = inputs[-1]
    U_tilde = jax.vmap(model)(U1).squeeze()
    U_tilde = U_tilde.reshape(U1.shape[0], 2, 8, 8)
    D = Dirac_Matrix(U1, kappa=0.276)
    M = Dirac_Matrix(U_tilde, kappa=0.276)

    U1 = U1.transpose((0, 2, 3, 1))
    v = jax.random.normal(
        key, (128,  *U1.shape), dtype=U1.dtype
    )
    v_pred = jax.vmap(HPD_opt, in_axes=(None, 0))(
        D, v
    )  # D.apply(D.apply(v), dagger=True)
    v_pred = jax.vmap(HPD_opt, in_axes=(None, 0))(
        M, v_pred
    )  # M.apply(M.apply(v_pred), dagger=True)
    loss = jnp.linalg.norm((v_pred - v).reshape(*v.shape[:2], -1), axis=-1)
    # loss = jnp.mean((v_pred - v)**2)
    return jnp.mean(loss)


def inverse_loss_multiU(model, inputs):
    U1 = inputs[0]
    key = inputs[-1]
    U_tilde = jax.vmap(model)(U1).squeeze()
    n_U = U_tilde.shape[-1] // 128
    U_tilde = U_tilde.reshape(U1.shape[0], n_U, 2, 8, 8)
    D = Dirac_Matrix(U1, kappa=0.276)

    Ms = []
    for i in range(n_U):
        M = Dirac_Matrix(U_tilde[:, i, ...], kappa=0.276)
        Ms.append(M)

    U1 = U1.transpose((0, 2, 3, 1))
    v = jax.random.normal(key, U1.shape, dtype=U1.dtype)
    v_pred = HPD_opt(D, v)  # D.apply(D.apply(v), dagger=True)
    for M in Ms:
        v_pred = HPD_opt(M, v_pred)  # M.apply(M.apply(v_pred), dagger=True)
    loss = jnp.linalg.norm((v_pred - v).reshape(U1.shape[0], -1), axis=-1)
    return jnp.mean(loss)


def condition_number_loss(model, U1, DD_mat):
    U1_flatten = U1.reshape(U1.shape[0], -1)[..., None]
    gammas = jax.vmap(model)(U1_flatten, DD_mat)
    Mopt = DiracGamma(U1.transpose((0, 3, 1, 2)), gammas)
    # need to construct the matrix of opt
    M_prime = construct_matrix(Mopt, U1.shape[0])
    M_prime = jnp.einsum(
        "bij, bjk -> bik", M_prime, M_prime.conj().transpose((0, 2, 1))
    )
    # jax.debug.print("Norm: {} | Submatrix: {}", jnp.linalg.norm(M_prime[0]), M_prime[0][:5, :5])
    M_inv = model.alpha * M_prime + jnp.identity(M_prime.shape[-1])[None, ...]
    precond_sys = jnp.einsum("bij, bjk -> bik", M_inv, DD_mat)
    cond_number = jnp.linalg.cond(precond_sys)
    return jnp.mean(cond_number)


def condition_number_loss_coo(model, U1, edges, DD_mat, adj_mat):
    _, M = jax.vmap(model, in_axes=(0, 0, None))(U1, edges, adj_mat)
    M = M.squeeze()
    batch_idx = jnp.broadcast_to(
        adj_mat.indices, (M.shape[0],) + adj_mat.indices.shape
    )
    M = sparse.BCOO((M, batch_idx), shape=(M.shape[0],) + adj_mat.shape)
    M = M.todense()

    MM = M @ M.conj().transpose((0, 2, 1))
    M_inv = model.alpha * MM + jnp.identity(MM.shape[-1])[None, ...]
    precond_sys = M_inv @ DD_mat
    cond_number = jnp.linalg.cond(precond_sys)
    return jnp.mean(cond_number)


def construct_matrix(opt, B, n=128):
    identity = jnp.identity(n)
    B_identity = jnp.repeat(identity[None, ...], B, axis=0)
    # X = jnp.sqrt(n // 2).astype(int).item()
    # T = jnp.sqrt(n // 2).astype(int).item()
    columns = []
    for i in range(B_identity.shape[1]):
        e_i = B_identity[:, :, i]
        e_i = e_i.reshape(B, 8, 8, 2)
        columns.append(
            opt.apply(e_i, dagger=False)
        )  # dagger False, need to mulitply with its conjugate transpose
    M = jnp.stack(columns, axis=1).reshape(B, n, n)
    return M


def condition_number_loss_with_mask(model, U1, DD, mask):
    nnzL = jax.vmap(model)(U1, DD).squeeze()
    L = jnp.zeros_like(DD)
    L = L.at[mask].set(nnzL.flatten())
    # L = jnp.where((DD != 0.0), nnzL.flatten(), 0.0)
    L = model.alpha * L + jnp.eye(L.shape[-1])
    LL_t = jnp.matmul(L, L.conj().transpose((0, 2, 1)))
    precond_sys = jnp.matmul(LL_t, DD)
    cond_number = jnp.linalg.cond(precond_sys)
    return jnp.mean(cond_number)


if __name__ == "__main__":

    class Model:
        def __init__(self, scale):
            self.alpha = scale

        def __call__(self, U1):
            return jnp.ones((128), dtype=jnp.complex64)

    dummy_model = Model(0.4)

    key = jax.random.PRNGKey(0)
    B = 10
    U = jax.random.normal(key, (B, 2, 8, 8), dtype=jnp.complex64)
    inputs = (U, U, U, key)
    loss = inverse_loss(dummy_model, inputs)

    print(loss)
