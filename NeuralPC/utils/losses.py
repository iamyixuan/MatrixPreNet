import torch
import numpy as np
from jax.scipy.sparse.linalg import cg
import jax.numpy as jnp
import jax
from jax import jit
from functools import partial


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

#@partial(jit, static_argnums=(0, 5))
def PCG_loss(NN, U1, b, kappa, steps, operator):
    """
    NN: the neural network, acting as the preconditioning operator.
    U1: gauge configuration of shape (b, 2, L, L).
    steps: total steps for PCG to run.
    operator: the operator of the original system.
    """
    print(U1.shape)
    # NN_vmap = jax.vmap(NN) # use this with equinox Module
    def runPCG(operator, b, precond=NN):
        x = jnp.ones(b.shape).astype(b.dtype)
        print( b.dtype, x.dtype)
        x_sol = cg(A=operator, b=b, x0=x, M=precond, maxiter=steps)
        return x_sol

    # fix the iteration step, calculate the residual b - Ax_sol and minimize the squared value.

    opt = partial(operator, U1=U1, kappa=kappa)
    x_sol, _ = runPCG(opt, b=b)
    residual = b - opt(x_sol)
    return jnp.mean(jnp.abs(residual)**2)

def testLoss(NN, x):
    NN_v = jax.vmap(NN)
    return jnp.mean(jnp.abs(NN_v(x))**2)


# @jax.jit
# def BatchPCGLoss(NN, U1, b, kappa, steps, operator):
#     batchResidual = jax.vmap(PCG_loss, in_axes=[None, 1, 2, None, None])(
#         NN, U1, b, kappa, steps, operator
#     )
#     return jnp.mean(batchResidual**2)
