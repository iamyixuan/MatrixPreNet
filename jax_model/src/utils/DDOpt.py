import jax
import jax.numpy as jnp
import numpy as np


def gamma_factory(n_dim, incl_g5=False):
    """A function to generate the 4x4 Wick-rotated gamma matrices.
    Arguments:
        n_dim: number of dimensions. Currently only 2 and
            4 are implemented.
        incl_g5: returns g5 as the final matrix.
    Returns:
        gammas: list of gamma matrices. In the 4D case,
            returns in the format [γ_x, γ_y, γ_z, γ_t]
                i.e. [γ_1, γ_2, γ_3, γ_0], contrary to convention.
            In the 2D case, returns the two 2D gamma matrices:
                [γ_1, γ_0]
                where γ_0 = i σ_x  // γ_0 = -i σ_y

    """
    # generate the sigma matrices (building blocks of gamma matrices).
    # sigma_mu = (I, sigma_x, sigma_y, sigma_z)
    paulis = [
        np.array([[1, 0], [0, 1]], dtype=np.cdouble),
        np.array([[0, 1], [1, 0]], dtype=np.cdouble),
        np.array([[0, -1j], [1j, 0]], dtype=np.cdouble),
        np.array([[1, 0], [0, -1]], dtype=np.cdouble),
    ]

    if n_dim == 4:
        # Weyl/chiral basis 0,1,2,3
        zeros_2x2 = np.zeros((2, 2), dtype=np.cdouble)
        gammas = np.array(
            [
                np.vstack(
                    (
                        np.hstack((zeros_2x2, sigma)),
                        np.hstack(((1 if mu == 0 else -1) * sigma, zeros_2x2)),
                    )
                )
                for mu, sigma in enumerate(paulis)
            ]
        )  # ^mu ^alpha _beta
        sigma = np.array(
            [
                [
                    0.5j
                    * (
                        np.dot(gammas[mu], gammas[nu])
                        - np.dot(gammas[nu], gammas[mu])
                    )
                    for nu in range(4)
                ]
                for mu in range(4)
            ]
        )

        # TO EUCLIDEAN: txyz -> xyztau
        # match conventions from https://en.wikipedia.org/wiki/Gamma_matrices#Chiral_representation
        gammas = [1j * g for g in gammas[1:]] + [gammas[0]]
        gammas[1] *= -1  # match conventions from QDP manual

        if incl_g5:
            gamma5 = gammas[0] @ gammas[1] @ gammas[2] @ gammas[3]
            return gammas + [gamma5]
        else:
            return gammas

        return gammas
    elif n_dim == 2:
        gammas = [paulis[1], paulis[2]]
        if incl_g5:
            gamma5 = 1j * (gammas[1] @ gammas[0])
            return gammas + [gamma5]
        else:
            return gammas

    else:
        raise NotImplementedError(
            "Only 2D and 4D gamma matrices are currently implemented"
        )


class Dirac_Matrix:
    def __init__(self, U, kappa):
        self.n_dim = 2
        self.time_index = 2

        self.U = U
        self.kappa = kappa

        # assert jnp.iscomplexobj(U)
        # assert len(U.shape) == 4
        # assert U.shape[1] == self.n_dim

        self.lattice_shape = U.shape[2:]
        self.gammas = jnp.stack(
            gamma_factory(n_dim=self.n_dim, incl_g5=False), axis=0
        )
        self.gammas = jnp.array(self.gammas).astype(U.dtype)
        self.gamma5 = 1j * jnp.dot(self.gammas[0], self.gammas[1])
        self.identity = jnp.eye(self.n_dim, dtype=U.dtype)

    def apply(self, x, dagger=False, gamma5=False):
        x = jnp.asarray(x, dtype=self.U.dtype)
        x = self._normalize_vector_shape(x)

        # if dagger:
        #     x = jnp.einsum("ij, ...j->...i", self.gamma5, x)
        def true_x(_):
            return jnp.einsum("ij, ...j->...i", self.gamma5, x)

        def false_x(_):
            return x

        x = jax.lax.cond(dagger, true_x, false_x, None)

        padded_x, slice_idx = self._antiperiodic_pad_vector(x)
        H = jnp.zeros_like(x)
        for mu in range(self.n_dim):
            forward_x = jnp.roll(padded_x, shift=-1, axis=1 + mu)
            forward_x = forward_x[slice_idx]
            forward_U = self.U[:, mu, ..., jnp.newaxis]
            # forward_U = self.U[:, mu, ...].reshape(list(self.U.shape[:-2]) + [1])
            forward_x = forward_U * forward_x

            backward_x = jnp.roll(padded_x, shift=1, axis=1 + mu)
            backward_x = backward_x[slice_idx]
            backward_U = jnp.roll(self.U[:, mu], shift=1, axis=1 + mu).conj()
            backward_U = backward_U[..., jnp.newaxis]
            # backward_U = backward_U.reshape(list(backward_U.shape[:-2]) + [1])
            backward_x = backward_U * backward_x

            H = H + jnp.einsum(
                "ij, ...j->...i", self.identity - self.gammas[mu], forward_x
            )
            H = H + jnp.einsum(
                "ij, ...j->...i", self.identity + self.gammas[mu], backward_x
            )

        y = x - (self.kappa * H)

        # if dagger:
        #     y = jnp.einsum("ij, ...j->...i", self.gamma5, y)
        def true_fun(_):
            return jnp.einsum("ij, ...j->...i", self.gamma5, y)

        def false_fun(_):
            return y

        y = jax.lax.cond(dagger, true_fun, false_fun, None)
        return y

    def _normalize_vector_shape(self, x):
        # assert len(x.shape) in [3, 4], f'unknown vector shape {x.shape}'
        # if len(x.shape) == 3:
        #     x = x.reshape((1,) + x.shape)  # dummy batch dimension
        # def true_branch(_):
        #     return x.reshape((1,) + x.shape)

        # def false_branch(_):
        #     return x

        # x = jax.lax.cond(x.ndim == 3, true_branch, false_branch, None)
        # assert x.shape[-1] == self.n_dim
        # assert x.shape[1:-1] == self.lattice_shape
        return x

    def _antiperiodic_pad_vector(self, x):
        padded_x = x.copy()

        s1 = [slice(None)] * len(x.shape)
        s1[self.time_index] = slice(0, 1)
        forward_pad = -padded_x[tuple(s1)]  # Convert list to tuple here

        s2 = [slice(None)] * len(x.shape)
        s2[self.time_index] = slice(
            x.shape[self.time_index] - 1, x.shape[self.time_index]
        )
        backward_pad = -padded_x[tuple(s2)]  # Convert list to tuple here

        padded_x = jnp.concatenate(
            [backward_pad, padded_x, forward_pad], axis=self.time_index
        )

        # Slice indices needed to recover the unpadded vector
        slice_idx = [slice(None)] * len(padded_x.shape)
        slice_idx[self.time_index] = slice(1, -1)

        # assert jnp.allclose(padded_x[tuple(slice_idx)], x), 'invalid padding'  # Convert list to tuple here
        return (padded_x, tuple(slice_idx))


class DiracGamma:
    def __init__(self, U, gammas, kappa=0.276):
        """
        gammas: the network output of shape (B, 2 * n_multiplies, 2, 2)
        """
        self.n_dim = 2
        self.time_index = 2

        self.U = U
        self.kappa = kappa

        self.lattice_shape = U.shape[2:]
        self.gammas = gammas.reshape(
            gammas.shape[0], gammas.shape[1] // 2, 2, 2, 2
        )  # shape (B, n_multiplies, 2, 2, 2)

        # TODO!! with multiple gamma matrices, we need to modify the gamma5 later
        self.gamma5 = 1j * jnp.einsum(
            "bnij, bnjk->bnik", self.gammas[:, :, 0], self.gammas[:, :, 1]
        )

        self.identity = jnp.eye(self.n_dim, dtype=U.dtype)
        self.identity = jnp.repeat(
            self.identity[None, ...], U.shape[0], axis=0
        )

    def apply(self, x, dagger=False, gamma5=False):
        x = jnp.asarray(x, dtype=self.U.dtype)
        x = self._normalize_vector_shape(x)

        def true_x(_):
            return jnp.einsum("bnij, b...j->b...i", self.gamma5, x)

        def false_x(_):
            return x

        x = jax.lax.cond(dagger, true_x, false_x, None)

        padded_x, slice_idx = self._antiperiodic_pad_vector(x)
        H = jnp.zeros_like(x)
        for mu in range(self.n_dim):
            forward_x = jnp.roll(padded_x, shift=-1, axis=1 + mu)
            forward_x = forward_x[slice_idx]
            forward_U = self.U[:, mu, ..., jnp.newaxis]
            forward_x = forward_U * forward_x

            backward_x = jnp.roll(padded_x, shift=1, axis=1 + mu)
            backward_x = backward_x[slice_idx]
            backward_U = jnp.roll(self.U[:, mu], shift=1, axis=1 + mu).conj()
            backward_U = backward_U[..., jnp.newaxis]
            backward_x = backward_U * backward_x

            # we can add an inner loop here over multiple (multiply of n_dim) gamma matrices
            for j in range(self.gammas.shape[1]):
                H = H + jnp.einsum(
                    "bij, b...j->b...i",
                    self.identity - self.gammas[:, j, mu],
                    forward_x,
                )
                H = H + jnp.einsum(
                    "bij, b...j->b...i",
                    self.identity + self.gammas[:, j, mu],
                    backward_x,
                )

        y = x - (self.kappa * H)

        def true_fun(_):
            return jnp.einsum("bnij, b...j->b...i", self.gamma5, y)

        def false_fun(_):
            return y

        y = jax.lax.cond(dagger, true_fun, false_fun, None)
        return y

    def _normalize_vector_shape(self, x):
        return x

    def _antiperiodic_pad_vector(self, x):
        padded_x = x.copy()

        s1 = [slice(None)] * len(x.shape)
        s1[self.time_index] = slice(0, 1)
        forward_pad = -padded_x[tuple(s1)]  # Convert list to tuple here

        s2 = [slice(None)] * len(x.shape)
        s2[self.time_index] = slice(
            x.shape[self.time_index] - 1, x.shape[self.time_index]
        )
        backward_pad = -padded_x[tuple(s2)]  # Convert list to tuple here

        padded_x = jnp.concatenate(
            [backward_pad, padded_x, forward_pad], axis=self.time_index
        )

        # Slice indices needed to recover the unpadded vector
        slice_idx = [slice(None)] * len(padded_x.shape)
        slice_idx[self.time_index] = slice(1, -1)

        # assert jnp.allclose(padded_x[tuple(slice_idx)], x), 'invalid padding'  # Convert list to tuple here
        return (padded_x, tuple(slice_idx))


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    B = 10
    U = jax.random.normal(key, (B, 2, 8, 8), dtype=jnp.complex64)
    # gammas = jax.random.normal(key, (B, 3, 2, 2), dtype=jnp.complex64)
    gammas = jnp.stack(
        gamma_factory(n_dim=2, incl_g5=False), axis=0, dtype=jnp.complex64
    )
    gammas = jnp.repeat(gammas[None, ...], B, axis=0)
    x = jax.random.normal(key, (B, 8, 8, 2), dtype=jnp.complex64)
    opt = DiracGamma(U, gammas)

    out = opt.apply(x, dagger=True)

    opt_true = Dirac_Matrix(U, 0.276)
    out_true = opt_true.apply(x, dagger=True)

    print(jnp.allclose(out, out_true))

    # the shape looks good, now we need a sanity check
    # to know if it really works with the real U1 field
    # and the gammas
