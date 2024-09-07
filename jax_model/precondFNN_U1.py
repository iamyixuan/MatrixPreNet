import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch
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
            self.U1 = U1.reshape(U1.shape[0], -1)
            self.DD = DD[train_idx]
        elif mode == "val":
            U1 = U1[val_idx]
            self.U1 = U1.reshape(U1.shape[0], -1)
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

    def __init__(
        self,
        key,
        in_dim,
        out_dim,
        activation,
        layer_sizes=[1024] * 3,
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

        self.scale = jnp.array([0.0])

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def condition_number_loss(model, U1, DD, mask):
    nnzL = jax.vmap(model)(U1).squeeze()
    L = jnp.zeros_like(DD)
    L = L.at[mask].set(nnzL.flatten())
    # L = jnp.where((DD != 0.0), nnzL.flatten(), 0.0)
    L = model.scale * L + jnp.eye(L.shape[-1])
    LL_t = jnp.matmul(L, L.conj().transpose((0, 2, 1)))
    precond_sys = jnp.matmul(LL_t, DD)
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
        loss_fn = condition_number_loss
    else:
        raise NotImplementedError

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def update_step(model, U1, DD, mask, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, U1, DD, mask)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    best_loss = 1e6
    for _ in range(configs["num_epochs"]):
        running_loss = 0.0
        for i, (U1, DD, tmask) in enumerate(trainloader):
            U1 = jnp.asarray(U1)
            DD = jnp.asarray(DD)
            tmask = jnp.nonzero(jnp.array(tmask))
            model, opt_state, loss = update_step(
                model, U1, DD, tmask, opt_state
            )
            running_loss += loss

        for U1_val, DD_val, vmask in valloader:
            U1_val = jnp.asarray(U1_val)
            DD_val = jnp.asarray(DD_val)

            vmask = jnp.nonzero(jnp.array(vmask))
            val_loss = loss_fn(model, U1_val, DD_val, vmask)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

        print(f"train Loss: {running_loss / (i + 1)}")
        print(f"scale: {model.scale}")
        print(f"val Loss: {val_loss}")

        logger.info(
            f"train Loss: {running_loss / (i + 1)}, val Loss: {val_loss}, scale: {model.scale.item()}"
        )

    return best_model


def main(data_path):
    logging.basicConfig(
        filename="./logs/U1_FNN_M/log.txt", level=logging.INFO, filemode="w"
    )

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    model = PrecondFNN(
        key=subkey,
        in_dim=128,
        out_dim=1792,
        activation=eqx.nn.PReLU(),
        layer_sizes=[1024] * 3,
    )

    config = {"num_epochs": 100, "batch_size": 128, "optim": optax.adam(1e-4)}
    # data_path = (
    #     "/home/seswar/Desktop/Academics/Code/nnprecond/datasets/DD_matrices.pt"
    # )
    trainset = U1DDDataset(data_path, mode="train")
    valset = U1DDDataset(data_path, mode="val")

    trainloader = DataLoader(trainset, batch_size=config["batch_size"])
    valloader = DataLoader(valset, batch_size=valset.__len__())

    model = train(
        model,
        trainloader,
        valloader,
        config["optim"],
        "conditionNumber",
        config,
    )
    eqx.tree_serialise_leaves("./logs/U1_FNN_M/model.eqx", model)


if __name__ == "__main__":
    data_path = "/Users/yixuan.sun/Documents/projects/Preconditioners/MatrixPreNet/data/U1_DD_matrices.pt"
    main(data_path)
