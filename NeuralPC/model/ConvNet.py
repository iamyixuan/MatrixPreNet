import torch
import torch.nn as nn


class LinearCNN(nn.Module):
    def __init__(self, ch_sizes, kernel_size) -> None:
        super(LinearCNN, self).__init__()
        self.layers = nn.ModuleList()

        for k in range(len(ch_sizes) - 1):
            self.layers.append(nn.Conv1d(ch_sizes[k], ch_sizes[k+1], kernel_size, stride=1, padding=int((kernel_size - 1)/2), bias=False)) 

    def forward(self, x_real, x_imag):
        x = torch.stack([x_real, x_imag], dim=1)

        for layer in self.layers:
            x = layer(x) 

        out = torch.cat([x[:, 0, :], x[:, 1, :]], dim=1)
        return out



