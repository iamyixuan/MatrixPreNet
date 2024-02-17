import torch
import torch.nn as nn


class LinearNN(nn.Module):
    def __init__(self, layer_sizes) -> None:
        super(LinearNN, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[k], layer_sizes[k+1], bias=False))
    def forward(self, r_real, r_imag):
        x = torch.cat([r_real, r_imag], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x