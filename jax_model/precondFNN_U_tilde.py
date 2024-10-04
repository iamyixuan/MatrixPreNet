import logging
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch
from src.utils.DDOpt import Dirac_Matrix
from src.utils.losses import (construct_matrix, inverse_loss,
                              inverse_loss_multiU)
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class U1DDDataset(Dataset):
    def __init__(self, datapath, mode, batch_size=None):
        self.mode = mode
        data = torch.load(datapath)
        U1 = data["U1"].numpy()
        DD = data["DD_mat"].numpy()
        self.mask = DD[0, :, :] != 0.0
        train_idx, val_idx = self.split_idx(U1.shape[0], 0.8)
        if mode == "train":
            U1 = U1[train_idx]
            self.U1 = U1  # .reshape(U1.shape[0], -1)
            self.DD = DD[train_idx]
        elif mode == "val":
            U1 = U1[val_idx]
            self.U1 = U1  # .reshape(U1.shape[0], -1)
            self.DD = DD[val_idx]

        print(
            "Initializing dataset - U1 {} DD {}".format(
                self.U1.shape, self.DD.shape
            )
        )

    def split_idx(self, n, split):
        idx = jax.random.permutation(jax.random.PRNGKey(0), n)
        return idx[: int(n * split)], idx[int(n * split) :]

    def __len__(self):
        return self.U1.shape[0]

    def __getitem__(self, idx):
        return self.U1[idx], self.DD[idx], self.mask


class ComplexLinear(eqx.Module):
    fc_layer: eqx.nn.Linear

    def __init__(self, *args, **kwargs):
        self.fc_layer = eqx.nn.Linear(*args, **kwargs)

    def __call__(self, x):
        x_real = x.real
        x_imag = x.imag
        return self.fc_layer(x_real) + 1j * self.fc_layer(x_imag)


class PrecondFNN(eqx.Module):
    layers: list
    scale: jnp.ndarray
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key,
        in_dim,
        out_dim,
        activation,
        dropout_rate=0.1,
        layer_sizes=[256] * 3,
    ):
        layer_sizes = [in_dim] + layer_sizes + [out_dim]
        nlayers = len(layer_sizes)
        keys = jax.random.split(key, nlayers - 1)

        # Create the params
        self.layers = []
        self.scale = jnp.array([0.0])

        for k in range(nlayers - 2):
            self.layers.append(
                ComplexLinear(
                    in_features=layer_sizes[k],
                    out_features=layer_sizes[k + 1],
                    key=keys[k],
                )
            )
            self.layers.append(activation)

        # No activation for last layer
        self.layers.append(
            ComplexLinear(
                in_features=layer_sizes[-2],
                out_features=layer_sizes[-1],
                key=keys[-1],
            )
        )

        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.scale = jnp.array([0.0])

    def __call__(self, x):
        x = x.flatten()
        for i, layer in enumerate(self.layers):
            x = self.dropout(x, key=jax.random.PRNGKey(i))
            x = layer(x)

        return x


def condition_number_loss_tilde(model, inputs):
    U1, DD, mask, _ = inputs
    U_tilde = jax.vmap(model)(U1).squeeze()
    U_tilde = U_tilde.reshape(U1.shape[0], 2, 8, 8)
    M = Dirac_Matrix(U_tilde, kappa=0.276)
    M = construct_matrix(
        M, B=U1.shape[0]
    )  # what if taking the lower triangular part of M
    M = model.scale * M + jnp.eye(M.shape[-1])
    MM = jnp.matmul(M, M.conj().transpose((0, 2, 1)))
    precond_sys = jnp.matmul(MM, DD)
    cond_number = jnp.linalg.cond(precond_sys)
    return jnp.mean(cond_number)


def train(
    model: PrecondFNN,
    trainloader: DataLoader,
    valloader: DataLoader,
    optim: optax.GradientTransformation,
    loss_name: str = "conditionNumber",
    configs: dict = None,
):
    print(f"Training with {loss_name} loss")
    if loss_name == "conditionNumber":
        loss_fn = condition_number_loss_tilde
    elif loss_name == "inverse":
        loss_fn = inverse_loss
    elif loss_name == "inverse_multiU":
        loss_fn = inverse_loss_multiU
    else:
        raise NotImplementedError

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def update_step(model, inputs, opt_state):
        model = eqx.nn.inference_mode(model, value=False)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, inputs)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def val_step(model, inputs):
        model = eqx.nn.inference_mode(model)
        return loss_fn(model, inputs)

    best_loss = 1e6
    patience = 0
    for _ in range(configs["num_epochs"]):
        running_loss = 0.0
        for i, (U1, DD, tmask) in enumerate(trainloader):
            key = jax.random.PRNGKey(i)
            k1, k2 = jax.random.split(key)
            U1 = jnp.asarray(U1)
            DD = jnp.asarray(DD)
            tmask = jnp.nonzero(jnp.array(tmask))
            inputs = (U1, DD, tmask, k1)
            model, opt_state, loss = update_step(model, inputs, opt_state)
            running_loss += loss

        for U1_val, DD_val, vmask in valloader:
            U1_val = jnp.asarray(U1_val)
            DD_val = jnp.asarray(DD_val)

            vmask = jnp.nonzero(jnp.array(vmask))
            inputs_val = (U1_val, DD_val, vmask, k2)
            val_loss = val_step(model, inputs_val)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            patience = 0
            print("Current best at epoch", _)
        else:
            patience += 1

        print(f"train Loss: {running_loss / (i + 1)}; val Loss: {val_loss}")
        # print(f"scale: {model.scale}")

        logger.info(
            f"train Loss: {running_loss / (i + 1)}, val Loss: {val_loss}, scale: {model.scale.item()}"
        )

        if patience > 50:
            break

    return best_model


def main(data_path):
    logname = "U1_FNN_U_tilde_full_inverse_loss_multiRandom128V"
    os.makedirs(f"./logs/{logname}", exist_ok=True)
    logging.basicConfig(
        filename=f"./logs/{logname}/log.txt", level=logging.INFO, filemode="w"
    )

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    model = PrecondFNN(
        key=subkey,
        in_dim=128,
        out_dim=128,
        activation=eqx.nn.PReLU(),
        layer_sizes=[1024] * 3,
    )

    config = {"num_epochs": 1000, "batch_size": 128, "optim": optax.adam(1e-4)}
    # data_path = (
    #     "/home/seswar/Desktop/Academics/Code/nnprecond/datasets/DD_matrices.pt"
    # )
    trainset = U1DDDataset(data_path, mode="train")
    valset = U1DDDataset(data_path, mode="val")
    logger.info(
        f"Train size {trainset.__len__()}, Val size {valset.__len__()}"
    )

    trainloader = DataLoader(trainset, batch_size=config["batch_size"])
    valloader = DataLoader(valset, batch_size=valset.__len__())

    model = train(
        model,
        trainloader,
        valloader,
        config["optim"],
        "inverse",
        config,
    )
    eqx.tree_serialise_leaves(f"./logs/{logname}/model.eqx", model)


if __name__ == "__main__":
    data_path = "../data/U1_DD_matrices.pt"
    main(data_path)
