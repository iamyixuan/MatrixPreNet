import torch
import torch.nn as nn


class ComplexConv2d(nn.Module):
    def __init__(self, *args, **kwrags) -> None:
        super(ComplexConv2d, self).__init__()
        self.conv = nn.Conv2d(*args, **kwrags)

    def forward(self, x):
        # check if x is complex
        assert x.is_complex(), "Input must be complex"
        return self.conv(x.real) + 1j * self.conv(x.imag)


class ComplexActivation(nn.Module):
    def __init__(self, activation) -> None:
        super(ComplexActivation, self).__init__()
        self.activation = activation

    def forward(self, x):
        # check if x is complex
        assert x.is_complex(), "Input must be complex"
        return self.activation(x.real) + 1j * self.activation(x.imag)


class ConvPrecondNet(nn.Module):
    """
    Convolutional network to produce lower triangular matrix L

    Args:
    in_ch: int, number of input channels
    out_ch: int, number of output channels
    kernel_size: int, size of the convolutional kernel
    """

    def __init__(
        self, in_ch, out_ch, kernel_size, num_layers=3, hidden_ch=32
    ) -> None:
        super(ConvPrecondNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            ComplexConv2d(in_ch, hidden_ch, kernel_size, padding="same")
        )
        self.layers.append(ComplexActivation(nn.PReLU()))
        for _ in range(num_layers - 2):
            self.layers.append(
                ComplexConv2d(
                    hidden_ch, hidden_ch, kernel_size, padding="same"
                )
            )
            self.layers.append(ComplexActivation(nn.PReLU()))
            self.layers.append(
                ComplexConv2d(hidden_ch, out_ch, kernel_size, padding="same")
            )

    def forward(self, x):
        """
        x: the DD matrix form with complex entries.
        """
        for layer in self.layers:
            x = layer(x)

        # take the lower part
        x = torch.tril(x.real) + 1j * torch.tril(x.imag)
        return x


class LinearCNN(nn.Module):
    def __init__(self, ch_sizes, kernel_size) -> None:
        super(LinearCNN, self).__init__()
        self.layers = nn.ModuleList()

        for k in range(len(ch_sizes) - 1):
            self.layers.append(
                nn.Conv1d(
                    ch_sizes[k],
                    ch_sizes[k + 1],
                    kernel_size,
                    stride=1,
                    padding=int((kernel_size - 1) / 2),
                    bias=False,
                )
            )

    def forward(self, x_real, x_imag):
        x = torch.stack([x_real, x_imag], dim=1)

        for layer in self.layers:
            x = layer(x)

        out = torch.cat([x[:, 0, :], x[:, 1, :]], dim=1)
        return out
