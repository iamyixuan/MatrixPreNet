import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch


class DataLoader:
    def __init__(self, datapath, mode, batch_size=None):
        self.mode = mode
        data = torch.load(datapath)
        U1 = data["U1"].numpy()
        DD = data["DD_mat"].numpy()
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
    L = jax.vmap(model)(x).squeeze()
    L = model.scale * L + jnp.eye(L.shape[-1])
    LL_t = jnp.matmul(L, L.conj().transpose((0, 2, 1)))
    precond_sys = jnp.matmul(LL_t, DD)
    cond_number = jnp.linalg.cond(precond_sys)
    return jnp.mean(cond_number)


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
    else:
        raise NotImplementedError

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def update_step(model, x, DD, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, DD)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for _ in range(configs["num_epochs"]):
        running_loss = 0.0
        for i, (x, DD) in enumerate(trainloader.load_data()):
            model, opt_state, loss = update_step(model, x, DD, opt_state)
            running_loss += loss

        for x_val, DD_val in valloader.load_data():
            val_loss = loss_fn(model, x_val, DD_val)

        print(f"train Loss: {running_loss / (i + 1)}")
        print(f"scale: {model.scale}")
        print(f"val Loss: {val_loss}")
    return model


def main():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    model = PrecondCNN(
        inch=2,
        outch=1,
        activation=eqx.nn.PReLU(),
        kernel_size=3,
        n_layers_gauge=10,
        n_layers_precond=5,
        hidden_dim=32,
        key=subkey,
    )

    config = {"num_epochs": 5, "batch_size": 128, "optim": optax.adam(1e-3)}
    data_path = "/Users/yixuan.sun/Documents/Projects/Preconditioners/MatrixPreNet/data/U1_DD_matrices.pt"

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
