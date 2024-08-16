import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch


class DataLoader:
    def __init__(self, datapath, mode, batch_size=None):
        self.mode = mode
        data     = torch.load(datapath)
        data     = data.numpy()
        data_nnz = data[data != 0.0].reshape(data.shape[0], -1)
        train_idx, val_idx = self.split_idx(len(data), 0.8)
        if mode == "train":
            self.data       = data[train_idx]
            self.data_nnz   = data_nnz[train_idx]
            self.batch_size = batch_size
            self.mask       = self.data[:batch_size] != 0.0
        elif mode == "val":
            self.data       = data[val_idx]
            self.data_nnz   = data_nnz[val_idx]
            self.batch_size = len(val_idx)
            self.mask       = self.data[:len(val_idx)] != 0.0

    def split_idx(self, n, split):
        idx = jax.random.permutation(jax.random.PRNGKey(0), n)
        return idx[: int(n * split)], idx[int(n * split) :]

    def load_data(self):
        for i in range(0, len(self.data), self.batch_size):
            yield self.data_nnz[i:i+self.batch_size], self.data[i:i+self.batch_size]


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
        layer_sizes=[1024]*3,
    ):
        layer_sizes = [in_dim] + layer_sizes + [out_dim]
        nlayers     = len(layer_sizes)
        keys = jax.random.split(key, nlayers-1)

        # Create the params
        self.layers = []
        self.scale  = jnp.array([0.0])

        for k in range(nlayers-2):
            self.layers.append(ComplexLinear(in_features=layer_sizes[k],
                out_features=layer_sizes[k+1], key=keys[k]))
            self.layers.append(activation)

        # No activation for last layer
        self.layers.append(ComplexLinear(in_features=layer_sizes[-2],
            out_features=layer_sizes[-1], key=keys[-1]))
        
        self.scale = jnp.array([0.0])

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def condition_number_loss(model, DD_nnz, DD, mask):
    nnzL = jax.vmap(model)(DD_nnz).squeeze()
    L = jnp.zeros_like(DD)
    L = L.at[jnp.array(mask)].set(nnzL.flatten())
    #L = jnp.where((DD != 0.0), nnzL.flatten(), 0.0)
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
    def update_step(model, DD_nnz, DD, mask, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, DD_nnz, DD, mask)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for _ in range(configs["num_epochs"]):
        running_loss = 0.0
        tmask = trainloader.mask.tolist()
        for i, (DD_nnz, DD) in enumerate(trainloader.load_data()):
            model, opt_state, loss = update_step(model, DD_nnz, DD, tmask, opt_state)
            running_loss += loss

        vmask = valloader.mask.tolist()
        for DD_nnz, DD_val in valloader.load_data():
            val_loss = loss_fn(model, DD_nnz, DD_val, vmask)

        print(f"train Loss: {running_loss / (i + 1)}")
        print(f"scale: {model.scale}")
        print(f"val Loss: {val_loss}")
    return model


def main():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    model = PrecondFNN(
        key=subkey,
        in_dim = 1792,
        out_dim = 1792,
        activation=eqx.nn.PReLU(),
        layer_sizes=[1024]*3,
    )

    config = {"num_epochs": 5, "batch_size": 128, "optim": optax.adam(1e-3)}
    data_path = "/home/seswar/Desktop/Academics/Code/nnprecond/datasets/DD_matrices.pt"

    trainloader = DataLoader(
        data_path, mode="train", batch_size=config["batch_size"]
    )
    valloader = DataLoader(data_path, mode="val")

    model = train(
        model,
        trainloader,
        valloader,
        config["optim"],
        "conditionNumber",
        config,
    )


if __name__ == "__main__":
    main()
