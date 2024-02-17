import torch
import torch.nn as nn


class NeuralPreconditioner(nn.Module):
    def __init__(self, basis_size, DDOpt):
        super().__init__()
        # Create basis of trainable vectors
        self.basis = nn.Parameter(torch.randn(basis_size, 8, 8, 2))
        self.DDOpt = DDOpt

        self.layers = nn.ModuleList()
        # create non-linear layers to output preconditioners
        pass

    def forward(self, x):
        # x is a random vector that is used to create the
        output_matrix = []
        for i in range(self.basis.shape[0]):
            output_matrix.append(self.DDOpt(self.basis[i]))

