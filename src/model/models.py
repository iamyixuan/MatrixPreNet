import torch
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, in_dim, out_dim, layer_sizes) -> None:
        super(FNN, self).__init__()
        self.layers = nn.ModuleList()
        layer_sizes = [in_dim] + layer_sizes + [out_dim]
        for k in range(len(layer_sizes) - 2):
            self.layers.append(nn.Linear(layer_sizes[k], layer_sizes[k+1]))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        