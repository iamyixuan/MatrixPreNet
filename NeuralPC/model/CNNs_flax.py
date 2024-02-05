import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from jax import random


def complex_kernel_init(rng, shape, dtype):
    fan_in = np.prod(shape) // shape[-1]
    x = random.normal(random.PRNGKey(rng[0]), shape) + 1j * random.normal(
        random.PRNGKey(rng[0]), shape
    )
    return x * (2 * fan_in) ** -0.5


def complex_bias_init(rng, shape, dtype):
    fan_in = np.prod(shape) // shape[-1]
    x = 0 + 1j * 0
    return x * (2 * fan_in) ** -0.5


class CNNEncoder(nn.Module):
    in_ch: int
    out_ch: int
    h_ch: int
    ker_size: tuple

    def setup(self):
        self.conv_1 = nn.Conv(
            self.in_ch,
            self.ker_size,
            strides=2,
        )
        self.conv_2 = nn.Conv(
            self.h_ch,
            self.ker_size,
            strides=2,
        )
        self.conv_3 = nn.Conv(
            self.out_ch,
            self.ker_size,
            strides=2,
        )

        self.dense1 = nn.Dense(512)
        self.dense2 = nn.Dense(256)
        self.dense3 = nn.Dense(128)
        self.act = nn.relu

    @nn.compact
    def __call__(self, x, train: bool):
        x = self.conv_1(x)
        x = self.act(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.conv_2(x)
        x = self.act(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.conv_3(x)
        x = self.act(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dense1(x)
        x = self.act(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.dense2(x)
        x = self.act(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.dense3(x)
        return x


class CNNDecoder(nn.Module):
    in_ch: int
    out_ch: int
    h_ch: int
    ker_size: tuple

    def setup(self):
        self.Tconv_1 = nn.ConvTranspose(
            self.out_ch,
            self.ker_size,
            strides=(2, 2),
        )
        self.Tconv_2 = nn.ConvTranspose(
            self.h_ch,
            self.ker_size,
            strides=(2, 2),
        )
        self.Tconv_3 = nn.ConvTranspose(
            16,
            self.ker_size,
            strides=(2, 2),
        )
        self.dense1 = nn.Dense(512)
        self.dense2 = nn.Dense(256)
        self.dense3 = nn.Dense(128)
        self.act = nn.relu

    @nn.compact
    def __call__(self, x, train: bool):
        x = self.dense3(x)
        x = self.act(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.dense2(x)
        x = self.act(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.dense1(x)
        x = self.act(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = x.reshape(x.shape[0], 1, 1, -1)
        x = self.Tconv_1(x)
        x = self.act(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.Tconv_2(x)
        x = self.act(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.Tconv_3(x)
        return x


class Encoder_Decoder(nn.Module):
    in_ch: int
    out_ch: int
    h_ch: int
    ker_size: tuple

    def setup(self):
        self.encoder = CNNEncoder(self.in_ch, self.out_ch, self.h_ch, self.ker_size)
        self.decoder = CNNDecoder(self.in_ch, self.out_ch, self.h_ch, self.ker_size)
        self.linear = nn.Dense(128 * 2)

    def __call__(self, x, train: bool):
        x = self.encoder(x, train=train)
        out = self.decoder(x, train=train)
        # out = self.linear(out.reshape(out.shape[0], -1))
        return out


class Conv4D(nn.Module):
    in_ch: int
    out_ch: int
    ker_size: tuple

    def setup(self):
        ker_shape = self.init_kernel(self.in_ch, self.out_ch, self.ker_size)
        self.kernel = self.param("kernel", nn.initializers.xavier_uniform(), ker_shape)

    def __call__(self, x):
        out = jax.lax.conv(x, self.kernel, (1, 1, 1, 1), "SAME")
        return out

    def init_kernel(self, in_ch, out_ch, ker_size):
        return (out_ch, in_ch, ker_size[0], ker_size[1], ker_size[2], ker_size[3])


class CNN4D(nn.Module):
    num_layers: int
    in_ch: int
    out_ch: int
    h_ch: int

    def setup(self):
        layers = []
        layers.append(Conv4D(self.in_ch, self.h_ch, (3, 3, 3, 3)))
        layers.append(nn.PReLU())

        for i in range(self.num_layers):
            layers.append(Conv4D(self.h_ch, self.h_ch, (3, 3, 3, 3)))
            layers.append(nn.PReLU())

        layers.append(Conv4D(self.h_ch, self.out_ch, (3, 3, 3, 3)))
        self.all_layers = layers

    def __call__(self, x):
        for l in self.all_layers:
            x = l(x)
        return x


class DiracPreconditionerCNN(nn.Module):
    L: int

    def setup(self):
        self.conv1 = nn.Conv(features=16, kernel_size=(3, 3), use_bias=True)
        self.conv2 = nn.Conv(features=32, kernel_size=(3, 3), use_bias=True)
        self.conv3 = nn.Conv(features=64, kernel_size=(3, 3), use_bias=True)
        self.dense = nn.Dense(features=self.L**4)
        self.final_conv = nn.Conv(features=4, kernel_size=(1, 1), use_bias=True)

    @nn.compact
    def __call__(self, x, train: bool):
        # First convolutional layer
        x = self.conv1(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        # Second convolutional layer
        x = self.conv2(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        # Third convolutional layer
        x = self.conv3(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        # Reshape and Dense layer
        x = x.reshape((x.shape[0], self.L * self.L, self.L * 64))
        x = self.dense(x)
        x = x.reshape((x.shape[0], self.L * self.L, self.L * self.L, 1))

        # Final convolution
        x = self.final_conv(x)
        return x


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10, 16, 16, 2))  # channel dim should be the last

    model = Encoder_Decoder(2, 4, 16, (3, 3))
    # print(model.tabulate(jax.random.key(0), x))

    params = model.init(key, x)

    # load data
    import numpy as np
    import optax

    optimizer = optax.adam(learning_rate=0.001)

    data = np.load(
        "../../../datasets/Dirac/precond_data/config.l16-N200-b2.0-k0.276-unquenched-test.x.npy"
    )
    data = jnp.asarray(data)
    data = jnp.transpose(data, [0, 2, 3, 1])

    @jax.jit
    def MSE(params, x, y):
        pred = model.apply(params, x)
        return jnp.mean((y - pred) ** 2)

    def split_idx(length, key):
        k = jax.random.PRNGKey(key)
        idx = jax.random.permutation(k, length)
        trainIdx = idx[: int(0.6 * length)]
        valIdx = idx[-int(0.4 * length) :]
        return trainIdx, valIdx

    trainIdx, valIdx = split_idx(data.shape[0], 42)
    trainData = data[trainIdx]
    valData = data[valIdx]

    opt_state = optimizer.init(params)
    loss_grad_fn = jax.value_and_grad(MSE)
    for i in range(5000):
        trainLoss, grads = loss_grad_fn(params, trainData, trainData)
        valLoss = MSE(params, valData, valData)
        # Do the learning - updating the params.
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if i % 10 == 0:
            print(
                "Loss step {}: ".format(i),
                trainLoss,
                "validation loss {}".format(valLoss),
            )
