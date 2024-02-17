"""
The objective is to learn the linear operator only from input output pairs
and the same linear mapping
"""

import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from NeuralPC.model.ConvNet import LinearCNN
from NeuralPC.model.LinearNet import LinearNN
from NeuralPC.utils.data import LinearMapData, split_dataset

# instantiate the model
n_feat = 128

# model = LinearNN([128*2, 128, 128, n_feat*2])
model = LinearCNN([2, 16, 16, 16, 2], 7)
model.double()

# create data loader
data = np.load("../../datasets/Dirac/U1_0-n1000_rz.npz")
(train_x, train_y), (val_x, val_y) = split_dataset(x=data["r"], y=data["z"])
train_data = LinearMapData(train_x, train_y)
val_data = LinearMapData(val_x, val_y)

TrainLoader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
)

ValLoader = DataLoader(
    val_data,
    batch_size=val_data.__len__(),
    shuffle=False,
)

# training loop
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
# Number of epochs
num_epochs = 300

# Training loop
log = {"train_loss": [], "val_loss": []}
for epoch in range(num_epochs):
    running_loss = []
    for data in TrainLoader:
        # Get the input data and target outputs
        r, z = data
        r_real, r_imag = r[0], r[1]
        z_real, z_imag = z[0], z[1]
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        z_pred = model(r_real, r_imag)

        loss_real = criterion(z_real, z_pred[:, :n_feat])
        loss_imag = criterion(z_imag, z_pred[:, n_feat:])
        loss = loss_real + loss_imag
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

    for data_val in ValLoader:
        r_val, z_val = data_val

        r_val_real, r_val_imag = r_val[0], r_val[1]
        z_val_real, z_val_imag = z_val[0], z_val[1]

        z_val_pred = model(r_val_real, r_val_imag)

        loss_real_val = criterion(z_val_real, z_val_pred[:, :n_feat])
        loss_imag_val = criterion(z_val_imag, z_val_pred[:, n_feat:])
        loss_val = loss_real_val + loss_imag_val

    train_loss = np.mean(running_loss)
    val_loss = loss_val.item()
    log["train_loss"].append(train_loss)
    log["val_loss"].append(val_loss)
    print(f"Epoch: {epoch + 1}, Loss: {train_loss:.3f} and {val_loss:.3f}")

with open("./log/log_U1_0-n1000_rz_CNN.pkl", "wb") as f:
    pickle.dump(log, f)
