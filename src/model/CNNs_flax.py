import jax
import jax.numpy as jnp
import flax.linen as nn


class CNNEncoder(nn.Module):
    in_ch: int
    out_ch: int
    h_ch: int
    ker_size: tuple

    def setup(self):
        self.conv_1 = nn.Conv(self.in_ch, self.ker_size, strides=2)
        self.conv_2 = nn.Conv(self.h_ch, self.ker_size, strides=2)
        self.conv_3 = nn.Conv(self.out_ch, self.ker_size, strides=2)

        self.Tconv_1 = nn.ConvTranspose(self.out_ch, self.ker_size, strides=2)
        self.Tconv_2 = nn.ConvTranspose(self.h_ch, self.ker_size, strides=2)
        self.Tconv_3 = nn.ConvTranspose(self.in_ch, self.ker_size, strides=2)

        self.act = nn.relu

    def __call__(self, x):
        x = self.conv_1(x)
        x = self.act(x)
        x = self.conv_2(x)
        x = self.act(x)
        x = self.conv_3(x)
        x = self.act(x)
        return x


class CNNDecoder(nn.Module):
    in_ch: int
    out_ch: int
    h_ch: int
    ker_size: tuple

    def setup(self):
        self.Tconv_1 = nn.ConvTranspose(self.out_ch, self.ker_size, strides=(2,2))
        self.Tconv_2 = nn.ConvTranspose(self.h_ch, self.ker_size, strides=(2,2))
        self.Tconv_3 = nn.ConvTranspose(self.in_ch, self.ker_size, strides=(2,2))
        self.act = nn.relu

    def __call__(self, x):
        x = self.Tconv_1(x)
        x = self.act(x)
        x = self.Tconv_2(x)
        x = self.act(x)
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

    def __call__(self, x):
        x = self.encoder(x)
        out = self.decoder(x)
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


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10,  16, 16, 2)) # channel dim should be the last
    
    model = Encoder_Decoder(2, 4, 16, (3,3))
    # print(model.tabulate(jax.random.key(0), x))

    params = model.init(key, x)

    # load data
    import numpy as np
    data = np.load('../../../datasets/Dirac/precond_data/config.l16-N200-b2.0-k0.276-unquenched-test.x.npy')
    data = jnp.asarray(data)
    data = jnp.transpose(data, [0, 2, 3, 1])
    def split_idx(length, key):
        k = jax.random.PRNGKey(key)
        idx = jax.random.permutation(k, length)
        trainIdx = idx[:int(.6*length)]
        valIdx = idx[-int(.4 * length):]
        return trainIdx, valIdx
    trainIdx, valIdx = split_idx(data.shape[0], 42)
    trainData = data[trainIdx]
    valData = data[valIdx]

    out = model.apply(params, trainData)
    print(out.shape)





