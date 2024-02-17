import pickle
from functools import partial

import jax
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from NeuralPC.model.CNNs_flax import Encoder_Decoder
from NeuralPC.train.TrainFlax import init_train_state, train_val
from NeuralPC.utils.data import create_dataLoader, split_idx
from NeuralPC.utils.dirac import DDOpt, DDOpt_torch


class MatrixNet(nn.Module):
    def __init__(self, in_features, out_features, num_h_layers):
        super(MatrixNet, self).__init__()
        self.layers = nn.ModuleList()
        self.input_layer = nn.Linear(in_features, 64)
        for i in range(num_h_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(64, 64))
        self.out_layer = nn.Linear(64, out_features)

    def forward(self, real, imag):
        x = torch.cat([real, imag], dim=1)
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        out = self.out_layer(x)  # entries for 128 by 128 by 2
        out = out.reshape(-1, 128, 128, 2)
        out_real = out[:, :, :, 0]
        out_imag = out[:, :, :, 1]
        out = torch.complex(out_real, out_imag)
        return out


class MAxLoss(nn.Module):
    def __init__(self):
        super(MAxLoss, self).__init__()

    def forward(self, v, pred, U1, kappa):
        """
        v: random vector of shape [B, 128]
        pred: M_inv of shape (B, 128, 128) complex valued
        U1: U1 field
        kappa: scalar
        """
        Av = DDOpt_torch(v.reshape(-1, 8, 8, 2), U1, 0.276)  # change kappa later
        M_invAv = torch.einsum("bij, bj -> bi", pred, Av.reshape(Av.shape[0], -1))
        residual = M_invAv - v
        return torch.mean(residual.real**2) + torch.mean(residual.imag**2)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.real = nn.Linear(in_features, out_features)
        self.imag = nn.Linear(in_features, out_features)

    def forward(self, real_input, imag_input):
        # Split the complex input into real and imaginary components

        # Compute the real and imaginary components of the output
        real_output = self.real(real_input) - self.imag(imag_input)
        imag_output = self.real(imag_input) + self.imag(real_input)
        return real_output, imag_output


class MyNetwork(nn.Module):
    def __init__(self, r_dim, U_dim, z_dim):
        super(MyNetwork, self).__init__()

        # Non-linear in U1
        self.U_linear = ComplexLinear(U_dim, 64)
        self.U_linear2 = ComplexLinear(64, 32)
        self.U_linear3 = ComplexLinear(32, 32)
        self.U_linear4 = ComplexLinear(32, 32)
        self.U_linear5 = ComplexLinear(32, 32)
        self.U_linear6 = ComplexLinear(32, 32)

        self.relu = nn.ReLU()

        self.complex_linear = ComplexLinear(r_dim + 32, z_dim)

    def forward(self, r_real, r_imag, U_real, U_imag):
        # Process U
        Up_real, Up_imag = self.U_linear(U_real, U_imag)
        Up_real = self.relu(Up_real)
        Up_imag = self.relu(Up_imag)
        Up_real, Up_imag = self.U_linear2(Up_real, Up_imag)
        Up_real = self.relu(Up_real)
        Up_imag = self.relu(Up_imag)
        Up_real, Up_imag = self.U_linear3(Up_real, Up_imag)
        Up_real = self.relu(Up_real)
        Up_imag = self.relu(Up_imag)
        Up_real, Up_imag = self.U_linear4(Up_real, Up_imag)
        Up_real = self.relu(Up_real)
        Up_imag = self.relu(Up_imag)
        Up_real, Up_imag = self.U_linear4(Up_real, Up_imag)
        Up_real = self.relu(Up_real)
        Up_imag = self.relu(Up_imag)
        Up_real, Up_imag = self.U_linear6(Up_real, Up_imag)

        # Combine r (real and imaginary) with processed U
        combined_real = torch.cat([r_real, Up_real], dim=1)
        combined_imag = torch.cat([r_imag, Up_imag], dim=1)
        # Linear transformation of r
        z = self.complex_linear(combined_real, combined_imag)
        return z


if __name__ == "__main__":
    # cwarnings.simplefilter('error')"
    jax.config.update("jax_enable_x64", True)
    torch.manual_seed(42)

    # U1 = np.load(
    #     "../../datasets/Dirac/precond_data/config.l8-N1600-b2.0-k0.276-unquenched.x.npy"
    # )
    # U1 = np.asarray(U1)
    # U1 = np.transpose(U1, [0, 2, 3, 1])  # shape B, X, T, 2

    with open(
        "../../datasets/Dirac/IC_pairs-10perU-config.l8-N1600-b2.0-k0.276.pkl", "rb"
    ) as f:
        data = pickle.load(f)

    # The data pairs are complex numbers!
    z = data["x"]
    r = data["v"]
    U1 = data["U1"]  # complex exponentiated

    U1 = np.asarray(U1).squeeze()
    z = np.asarray(z)
    r = np.asarray(r)

    print(U1.shape)

    trainIdx, valIdx = split_idx(U1.shape[0], 42)

    trainU1 = U1[trainIdx]
    valU1 = U1[valIdx]

    train_z = z[trainIdx]
    val_z = z[valIdx]

    train_r = r[trainIdx]
    val_r = r[valIdx]

    train_data = [trainU1, train_z, train_r]
    val_data = [valU1, val_z, val_r]

    TrainLoader = create_dataLoader(
        data=train_data,
        batchSize=32,
        kappa=0.276,
        shuffle=True,
        dataset="IC",
    )

    ValLoader = create_dataLoader(
        data=val_data,
        kappa=0.276,
        batchSize=256,
        shuffle=False,
        dataset="IC",
    )

    epochs = 500
    learning_rate = 1e-5
    h_ch = 16

    # model = MyNetwork(r_dim=128, z_dim=128, U_dim=128)
    model = MatrixNet(256, 128 * 128 * 2, 3)
    model.double()
    # criterion = nn.MSELoss()
    criterion = MAxLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(
        optimizer, "min", patience=30, verbose=True, factor=0.7, min_lr=1e-5
    )
    # Number of epochs
    num_epochs = 100

    # Training loop
    log = {"train_loss": [], "val_loss": []}
    for epoch in range(num_epochs):
        running_loss = []
        for data in TrainLoader:
            # Get the input data and target outputs
            U, r, z, kappa = data
            U_m = U.reshape(U.shape[0], -1)
            r_real, r_imag = r[0], r[1]
            z_real, z_imag = z[0], z[1]
            r = torch.complex(r_real, r_imag)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            # z_real_pred, z_imag_pred = model(r_real, r_imag, U.real, U.imag)
            M_inv = model(U_m.real, U_m.imag)
            loss = criterion(r, M_inv, U, kappa)

            # Compute loss separately for real and imaginary parts and combine
            # loss_real = criterion(z_real_pred, z_real)
            # loss_imag = criterion(z_imag_pred, z_imag)
            # loss = loss_real + loss_imag

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

        for data_val in ValLoader:
            U_val, r_val, z_val, kappa = data_val
            U_m_val = U_val.reshape(U_val.shape[0], -1)
            r_val = torch.complex(r_val[0], r_val[1])
            # r_real_val, r_imag_val = r_val[0], r_val[1]
            # z_real_val, z_imag_val = z_val[0], z_val[1]
            # z_real_pred_val, z_imag_pred_val = model(r_real_val, r_imag_val, U_val.real, U_val.imag)

            # Compute loss separately for real and imaginary parts and combine
            # loss_real_val = criterion(z_real_pred_val, z_real_val)
            # loss_imag_val = criterion(z_imag_pred_val, z_imag_val)
            # loss_val = loss_real_val + loss_imag_val

            M_inv_val = model(U_m_val.real, U_m_val.imag)
            loss_val = criterion(r_val, M_inv_val, U_val, kappa)

        # scheduler.step(loss_val)

        # if i % 100 == :  # print every 100 mini-batches
        train_loss = np.mean(running_loss)
        val_loss = loss_val.item()
        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)
        print(f"Epoch: {epoch + 1}, Loss: {train_loss:.3f} and {val_loss:.3f}")

    with open("./log/MinvAv_log.pkl", "wb") as f:
        pickle.dump(log, f)
    print("Finished Training")
