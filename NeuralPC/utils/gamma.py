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
                    * (np.dot(gammas[mu], gammas[nu]) - np.dot(gammas[nu], gammas[mu]))
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
