import numpy as np
import torch
import torch.nn as nn


class LinearInverse(nn.Module):
    def __init__(self, num_layers):
        super(LinearInverse, self).__init__()
        hidden_layer_size = 256
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_layer_size, hidden_layer_size, bias=False)
                for _ in range(num_layers)
            ]
        )
        self.input_layer = nn.Linear(128 * 2, hidden_layer_size, bias=False)
        self.final_layer = nn.Linear(hidden_layer_size, 128 * 2, bias=False)

    def forward(self, x):
        x_real, x_imag = x.real, x.imag
        x = torch.cat((x_real, x_imag), dim=1)

        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        x_real, x_imag = x[:, :128], x[:, 128:]
        x = torch.complex(x_real, x_imag)
        return x
