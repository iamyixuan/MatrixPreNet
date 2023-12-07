import torch
import numpy as np
from jax.scipy.sparse.linalg import cg
import jax.numpy as jnp
import jax
from jax import jit
from functools import partial
from ..utils.conjugate_gradient import solve
from ..model.linearOpt import linearConvOpt, linearOpt
from ..utils.dirac import DDOpt


class Losses:
    def __init__(self, loss_name) -> None:
        """
        loss: [MSE, MAE]
        """
        self.loss_name = loss_name

    def __call__(self):
        ls_fn = self.get_loss(self.loss_name)
        return ls_fn

    def get_lam(self, true, pred, adj_true):
        """
        true and pred have the shape of [num_data, timesteps, x_locs]
        adj_true has the shape of [num_data, timesteps, y_locs, x_locs]
        """
        init_lam = 2 * (true[:, -1, :] - pred[:, -1, :])  # shape [num_data, x_locs, 1]
        init_lam = torch.clip(init_lam, 0, np.inf)

        lam = [init_lam]
        for t in range(1, true.shape[-2]):
            lam_T_t = 2 * (true[:, -1 - t, :] - pred[:, -1 - t, :]) + torch.einsum(
                "bj, bjk -> bk", lam[-1], adj_true[:, -1 - t, :, :]
            )  # lam[-1] @ adj_true[:, -1-t, :, :]
            lam_T_t = torch.clip(lam_T_t, 0, np.inf)
            lam.append(lam_T_t)
        # the lamdbas are from the last timestep to the first, so we need to reverse it to return
        lam = torch.stack(lam[::-1])
        return torch.permute(lam, (1, 0, 2))

    def MSE(self, true, pred):
        loss = torch.pow(true - pred, 2)
        return torch.mean(loss)

    def LagrangianLoss(self, true, pred, adj):
        # implement the Lagrangian of the optimization problem.
        lam = self.get_lam(true, pred, adj)
        L = self.MSE(true, pred) + torch.mean(lam * (true - pred) ** 2)
        return L

    def MAE(self, true, pred):
        loss = torch.abs(true - pred)
        return torch.mean(loss)

    def get_loss(self, loss_name):
        if loss_name == "MSE":
            return self.MSE
        elif loss_name == "MAE":
            return self.MAE
        elif loss_name == "Lag":
            return self.LagrangianLoss
        else:
            raise Exception("Loss name not recognized!")


def solveWithOpt(U1, b, w, bias, steps):
    def linOpt(x):
        new_vect = linearOpt(x, w, bias)
        return new_vect

    def opt(x):
        return DDOpt(x, U1, kappa=0.276)

    x0 = jnp.zeros_like(b)
    state = solve(opt, b, x0, steps, M=linOpt)
    return state.x


vmap_solve = jax.vmap(solveWithOpt, in_axes=[0, 0, 0, 0, None], out_axes=0)


# @partial(jit, static_argnums=(0, 5))
def PCG_loss(
    params, batch_stats, model, U1, b, in_mat, kappa, steps, operator, train=False
):
    """
    params: NN weights and biases.
    batch_stats: batch stats related to batchnorm; can get from the Flax model.
    model: the NN model implemented in Flax.
    U1: gauge configuration of shape (b, 2, L, L). (angle only)
    b: RHS of the linear system; generated as a random vector.
    in_mat: reshaped U1 for NN input.
    kappa: partially determine the DD operator.
    steps: total steps for PCG to run.
    operator: the operator of the original system.
    """
    U1_batch = U1[:, jnp.newaxis, ...]
    b_batch = b[:, jnp.newaxis, ...]

    def runPCG(b):
        kernels, updates = model.apply(
            {"params": params, "batch_stats": batch_stats},
            in_mat,
            mutable=["batch_stats"],
            train=train,
        )
        # kernels = jnp.reshape(kernels, (kernels.shape[:-1] + (4, 4)))
        w, bias = kernels[:, :128], kernels[:, -128:]
        w = jnp.tile(jnp.eye(128, 128), (kernels.shape[0], 1, 1))
        x_sol = vmap_solve(U1_batch, b_batch, w, bias, steps)
        x_sol = x_sol[:, 0, ...]  # remove the dummy dim.
        return x_sol, updates

    # fix the iteration step, calculate the residual b - Ax_sol and minimize the squared value.
    opt = partial(operator, U1=U1, kappa=kappa)
    state, updates = runPCG(b=b)
    residual = b - opt(state)
    residual = residual.reshape(residual.shape[0], -1)
    norm = jnp.mean(jnp.linalg.norm(residual, axis=1))
    return norm, updates


def testLoss(NN, x):
    NN_v = jax.vmap(NN)
    return jnp.mean(jnp.abs(NN_v(x)) ** 2)
