import logging
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, datapath, mode, batch_size=None):
        self.mode = mode
        data = torch.load(datapath)
        U1 = data["U1"].numpy()
        DD = data["DD_mat"].numpy()
        self.DD_mask = DD[0] == 0
        train_idx, val_idx = self.split_idx(len(U1), 0.8)
        if mode == "train":
            self.U1 = U1[train_idx]
            self.DD = DD[train_idx]
            self.batch_size = batch_size
        elif mode == "val":
            self.U1 = U1[val_idx]
            self.DD = DD[val_idx]
            self.batch_size = len(val_idx)

    def split_idx(self, n, split):
        idx = jax.random.permutation(jax.random.PRNGKey(0), n)
        return idx[: int(n * split)], idx[int(n * split) :]

    def load_data(self):
        for i in range(0, len(self.U1), self.batch_size):
            yield self.U1[i : i + self.batch_size], self.DD[
                i : i + self.batch_size
            ]


class ComplexConv2d(eqx.Module):
    conv_layer: eqx.nn.Conv2d

    def __init__(self, *args, **kwargs):
        self.conv_layer = eqx.nn.Conv2d(*args, **kwargs)

    def __call__(self, x):
        x_real = x.real
        x_imag = x.imag
        return self.conv_layer(x_real) + 1j * self.conv_layer(x_imag)


class PrecondCNN(eqx.Module):
    gauge_layers: list
    precond_layers: list
    scale: jnp.ndarray

    def __init__(
        self,
        inch,
        outch,
        activation,
        kernel_size,
        n_layers_gauge,
        n_layers_precond,
        hidden_dim,
        key,
    ):
        keys = jax.random.split(key, n_layers_gauge + n_layers_precond)
        padding = int((kernel_size - 1) / 2)
        self.gauge_layers = []
        self.precond_layers = []
        self.scale = jnp.array([0.0])
        for i in range(n_layers_gauge):
            if i == 0:
                self.gauge_layers.append(
                    ComplexConv2d(
                        inch,
                        hidden_dim,
                        kernel_size,
                        padding=padding,
                        key=keys[i],
                    )
                )
            else:
                self.gauge_layers.append(
                    ComplexConv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size,
                        padding=padding,
                        key=keys[i],
                    )
                )
            self.gauge_layers.append(activation)

        for i in range(n_layers_precond):
            if i == n_layers_precond - 1:
                self.precond_layers.append(
                    ComplexConv2d(
                        hidden_dim,
                        outch,
                        kernel_size,
                        padding=padding,
                        key=keys[i + n_layers_gauge],
                    )
                )
            else:
                self.precond_layers.append(
                    ComplexConv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size,
                        padding=padding,
                        key=keys[i + n_layers_gauge],
                    )
                )
            if i < n_layers_precond - 1:
                self.precond_layers.append(activation)

    def __call__(self, x):
        for g_layer in self.gauge_layers:
            x = g_layer(x)

        w = x.shape[-1]
        x = jax.image.resize(x, (x.shape[0], 2 * w**2, 2 * w**2), "bilinear")

        for p_layer in self.precond_layers:
            x = p_layer(x)
        return x


def condition_number_loss(model, x, DD):
    # take the lower triangular part of the matrix
    L = jnp.tril(jax.vmap(model)(x).squeeze())
    L = model.scale * L + jnp.eye(L.shape[-1])
    LL_t = jnp.matmul(L, L.conj().transpose((0, 2, 1)))
    precond_sys = jnp.matmul(LL_t, DD)
    cond_number = jnp.linalg.cond(precond_sys)
    return jnp.mean(cond_number)


def inversion_loss(model, x, DD):
    L = jnp.tril(jax.vmap(model)(x).squeeze())
    L = model.scale * L + jnp.eye(L.shape[-1])
    LL_t = jnp.matmul(L, L.conj().transpose((0, 2, 1)))
    I_hat = jnp.matmul(LL_t, DD)
    return jnp.mean(jnp.linalg.norm(I_hat - jnp.eye(I_hat.shape[-1])))


def train(
    model: PrecondCNN,
    trainloader: DataLoader,
    valloader: DataLoader,
    optim: optax.GradientTransformation,
    loss_name: str = "conditionNumber",
    configs: dict = None,
):
    print(f"Training with {loss_name} loss")
    if loss_name == "conditionNumber":
        loss_fn = condition_number_loss
    elif loss_name == "inversion":
        loss_fn = inversion_loss
    else:
        raise NotImplementedError

    loss_fn = eqx.filter_jit(loss_fn)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def update_step(model, x, DD, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, DD)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def val_step(model, x, DD):
        return loss_fn(model, x, DD)

    best_loss = jnp.inf
    for _ in range(configs["num_epochs"]):
        running_loss = 0.0
        for i, (x, DD) in enumerate(trainloader.load_data()):
            model, opt_state, loss = update_step(model, x, DD, opt_state)
            running_loss += loss

        for x_val, DD_val in valloader.load_data():
            val_loss = val_step(model, x_val, DD_val)

        if val_loss < best_loss:
            best_loss = val_loss
            eqx.tree_serialise_leaves(
                f"./logs/{configs['log_dir']}/best_model.eqx", model
            )

        print(f"train Loss: {running_loss / (i + 1)}")
        print(f"scale: {model.scale}")
        print(f"val Loss: {val_loss}")
        logger.info(
            f"train Loss: {running_loss / (i + 1)}, val Loss: {val_loss}"
        )
        # print(f"Time taken: {time.time() - current_time}")
    return model


def main(configs):
    log_name = "{}_{}_{}_{}_{}".format(
        configs["n_layers_gauge"],
        configs["n_layers_precond"],
        configs["hidden_dim"],
        configs["batch_size"],
        configs["loss"],
    )
    if not os.path.exists(f"./logs/{log_name}"):
        os.makedirs(f"./logs/{log_name}")
    logging.basicConfig(
        filename=f"./logs/{log_name}/train_logs.txt",
        level=logging.INFO,
    )
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    model = PrecondCNN(
        inch=2,
        outch=1,
        activation=eqx.nn.PReLU(),
        kernel_size=3,
        n_layers_gauge=configs["n_layers_gauge"],
        n_layers_precond=configs["n_layers_precond"],
        hidden_dim=configs["hidden_dim"],
        key=subkey,
    )

    data_path = "../data/U1_DD_matrices.pt"

    trainloader = DataLoader(
        data_path, mode="train", batch_size=configs["batch_size"]
    )
    valloader = DataLoader(data_path, mode="val")

    model = train(
        model,
        trainloader,
        valloader,
        configs["optim"],
        configs["loss"],
        configs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_layers_gauge", type=int, default=10, help="Number of gauge layers"
    )
    parser.add_argument(
        "--n_layers_precond",
        type=int,
        default=5,
        help="Number of preconditioner layers",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=32, help="Hidden dimension"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--log_dir", type=str, default="test_run", help="Log directory"
    )
    parser.add_argument(
        "--loss", type=str, default="conditionNumber", help="Loss function"
    )
    args = parser.parse_args()

    configs = {
        "n_layers_gauge": args.n_layers_gauge,
        "n_layers_precond": args.n_layers_precond,
        "hidden_dim": args.hidden_dim,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "log_dir": "test_run",
        "optim": optax.adam(args.learning_rate),
        "loss": args.loss,
    }
    main(configs)
