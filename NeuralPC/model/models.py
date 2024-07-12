import torch
import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    def __init__(
        self, in_dim, out_dim, activation, layer_sizes=[1024] * 3
    ) -> None:
        super(FNN, self).__init__()
        self.layers = nn.ModuleList()
        layer_sizes = [in_dim] + layer_sizes + [out_dim]
        for k in range(len(layer_sizes) - 2):
            self.layers.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))
            # self.layers.append(nn.BatchNorm1d(layer_sizes[k + 1]))
            self.layers.append(activation)

        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        # self.layers.append(nn.BatchNorm1d(layer_sizes[-1]))
        # self.layers.append(nn.Softplus())
        self.scale = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x):
        for layer in self.layers:
            x_real = layer(x.real)
            x_imag = layer(x.imag)
            x = x_real + 1j * x_imag
        return x


class PrecondCNN(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        n_layers_gauge=3,
        n_layers_precond=3,
        kernel_size=3,
    ):
        super(PrecondCNN, self).__init__()
        self.gauge_layers = nn.ModuleList()
        self.precond_layers = nn.ModuleList()
        self.scale = nn.Parameter(torch.tensor(1e-3))

        for k in range(n_layers_gauge):
            if k == 0:
                self.gauge_layers.append(
                    nn.Conv2d(
                        in_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding="same",
                    )
                )
            else:
                self.gauge_layers.append(
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding="same",
                    )
                )

            self.gauge_layers.append(nn.BatchNorm2d(hidden_dim))
            self.gauge_layers.append(nn.ReLU())

        for k in range(n_layers_precond):
            self.precond_layers.append(
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding="same",
                )
            )
            self.precond_layers.append(nn.BatchNorm2d(hidden_dim))
            self.precond_layers.append(nn.ReLU())

        self.output = nn.Conv2d(
            hidden_dim, out_dim, kernel_size=kernel_size, padding="same"
        )

    def forward(self, x):
        scale_factor = x.size(-1)
        for gauge_layer in self.gauge_layers:
            x_real = gauge_layer(x.real)
            x_imag = gauge_layer(x.imag)
            x = x_real + 1j * x_imag

        x = nn.functional.interpolate(
            x.real, scale_factor=2 * scale_factor
        ) + 1j * nn.functional.interpolate(
            x.imag, scale_factor=2 * scale_factor
        )

        for precond_layer in self.precond_layers:
            x_real = precond_layer(x.real)
            x_imag = precond_layer(x.imag)
            x = x_real + 1j * x_imag

        out = self.output(x.real) + 1j * self.output(x.imag)
        out = torch.tril(out.squeeze())
        return out


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=64):
        super(CNNEncoder, self).__init__()

        # Encoder layers
        self.conv1 = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(
            64, latent_channels, kernel_size=3, stride=2, padding=1
        )
        self.bn3 = nn.BatchNorm2d(latent_channels)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.bn1(x)
        x = F.tanh(self.conv2(x))
        x = self.bn2(x)
        x = F.tanh(self.conv3(x))
        x = self.bn3(x)
        return x


class CNNDecoder(nn.Module):
    def __init__(self, out_channels=1, latent_channels=64):
        super(CNNDecoder, self).__init__()

        # Decoder layers
        self.deconv1 = nn.ConvTranspose2d(
            latent_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            32, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.tanh(self.deconv1(x))
        x = self.bn1(x)
        x = F.tanh(self.deconv2(x))
        x = self.bn2(x)
        x = self.deconv3(x)
        return x


class CNNEncoderDecoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, latent_channels=64):
        super(CNNEncoderDecoder, self).__init__()

        self.encoder = CNNEncoder(in_channels, latent_channels)
        self.decoder = CNNDecoder(out_channels, latent_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DiracCNN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, hid_ch=64, num_layers=8):
        super(DiracCNN, self).__init__()
        self.layers = nn.ModuleList()
        self.input = nn.Conv2d(in_ch, hid_ch)
        self.layers.append(self.input)


class RNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, n_layers=3):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = FNN(hidden_dim, out_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out
