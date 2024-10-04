import equinox as eqx
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn

from .DDOpt import Dirac_Matrix


def compute_condition_number(A):
    return jnp.linalg.cond(A)


def get_batch_matrix(f, b_size, v_size=128):
    """construct the matrix form of an operator
    Args:
        f: function that takes a batch of vectors and returns a batch of vectors
        v_size: size of the vector
    """
    identity = jnp.identity(v_size)
    identity = jnp.repeat(identity[None, ...], b_size, axis=0)
    M = jax.vmap(f, in_axes=-1, out_axes=-1)(identity)
    M = M.reshape(M.shape[0], v_size, v_size)
    return M


def load_model(configs, model, checkpoint):
    model = model(**configs)
    model = eqx.tree_deserialise_leaves(checkpoint + "model.eqx", model)
    model = eqx.nn.inference_mode(model)
    return model


def construct_matrix(opt, B, n=128):
    identity = jnp.identity(n)
    B_identity = jnp.repeat(identity[None, ...], B, axis=0)
    columns = []
    for i in range(B_identity.shape[1]):
        e_i = B_identity[:, :, i]
        e_i = e_i.reshape(B, 8, 8, 2)
        columns.append(
            opt(e_i)
        )  
    M = jnp.stack(columns, axis=1).reshape(B, n, n)
    return M


def construct_Dirac_Matrix(U1, kappa=0.276):
    if U1.shape[-3:] != (2, 8, 8):
        U1 = U1.reshape(U1.shape[0], 2, 8, 8)
    return Dirac_Matrix(U1, kappa=kappa)


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

if __name__ == "__main__":
    # test matrix constr
    def f(x):
        if x.shape[-3:] != (8, 8, 2):
            x = x.reshape(x.shape[0], 8, 8, 2)
        return x

    M = get_batch_matrix(f)
    print(M[0])
