"""
Build neural network as a kernel function
"""

import torch
import torch.nn as nn


class denseNN(nn.Module):
    def __init__(self, layer_sizes) -> None:
        super(denseNN, self).__init__()
        self.layers = nn.ModuleList()

        for k in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[k], layer_sizes[k+1], bias=False))
    def forward(self, x, U1):
        x = torch.cat([x, U1], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x

class CNN(nn.Module):
    def __init__(self, layer_sizes) -> None:
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()

        for k in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[k], layer_sizes[k+1], bias=False))

    def forward(self, x, U1):
        pass
        

class CholeskyKernel(nn.Module):
    def __init__(self, layer_sizes) -> None:
        super(CholeskyKernel, self).__init__()
        self.layers = nn.ModuleList()

        for k in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[k], layer_sizes[k+1]))
            if k < len(layer_sizes) - 1: # only non-linear activation in hidden layers
                self.layers.append(nn.ReLU())
    def forward(self, x, y, U1):
        # x = torch.ones([U1.shape[0],1]) * x
        # y = torch.ones([U1.shape[0], 1]) * y
        x = torch.cat([x, y, U1.real, U1.imag], dim=1)
        for layers in self.layers:
            x = layers(x)
        return x.squeeze()

'''
Now we construct the kernel matrix based on the 
kernel function approximated by the NN.
'''

class LinearOperator(nn.Module):
    def __init__(self,  layer_sizes) -> None:
        super(LinearOperator, self).__init__()

        self.kernel_func = CholeskyKernel(layer_sizes)
        
    def getKernelMat(self, v, U1):
        # v_size = v.shape[1]
        # G_mat = torch.zeros([v.shape[0], v_size, v_size])
        # for i in range(v_size):
        #     for j in range(0, i+1): # lower triangular
        #         G_mat[:, i, j] = self.kernel_func(i, j, U1)
        # G_mat_conjT = G_mat.conj().transpose(-2, -1)

        # K_mat = torch.bmm(G_mat, G_mat_conjT) # GG^T
        batch_size, v_size = v.shape[0], v.shape[1]
        indices = torch.tril_indices(v_size, v_size)  # Lower triangular indices
        num_indices = indices.shape[1]

        # Create grid of indices for lower triangular matrix
        i_indices = indices[0].unsqueeze(0).repeat(batch_size, 1)
        j_indices = indices[1].unsqueeze(0).repeat(batch_size, 1)

        # Prepare inputs for CholeskyKernel
        U1_expanded = U1.unsqueeze(1).expand(-1, num_indices, -1).reshape(-1, U1.shape[-1])
        i_indices_flat = i_indices.reshape(-1, 1)
        j_indices_flat = j_indices.reshape(-1, 1)

        # Compute all values with CholeskyKernel in a batched manner
        G_values_flat = self.kernel_func(i_indices_flat, j_indices_flat, U1_expanded)

        # Reshape and assign to G_mat
        G_mat = torch.zeros(batch_size, v_size, v_size, dtype=G_values_flat.dtype, device=G_values_flat.device)
        G_mat[:, indices[0], indices[1]] = G_values_flat.view(batch_size, num_indices)

        G_mat_conjT = G_mat.conj().transpose(-2, -1)
        K_mat = torch.bmm(G_mat, G_mat_conjT) # GG^T
        return K_mat

    def forward(self, v, U1):
        K_mat = self.getKernelMat(v, U1).to(torch.complex64)
        out = torch.einsum('bij, bj -> bi', K_mat, v)
        return out

        

if __name__ == '__main__':
    import pickle 
    import numpy as np
    from NeuralPC.utils.data import split_idx, create_dataLoader
    from NeuralPC.train import trainer
    import torch.optim as optim
    from torch.profiler import profile, record_function, ProfilerActivity



    with open("/Users/yixuan.sun/Documents/projects/Preconditioners/datasets/Dirac/IC_pairs-10perU-config.l8-N1600-b2.0-k0.276.pkl", "rb") as f:
        data = pickle.load(f)


    # The data pairs are complex numbers!
    z = data["x"]
    r = data["v"]
    U1 = data['U1'] # complex exponentiated


    U1 = np.asarray(U1).squeeze().astype(np.complex64)
    
    z = np.asarray(z).astype(np.complex64)
    r = np.asarray(r).astype(np.complex64)



    trainIdx, valIdx = split_idx(U1.shape[0], 42)
    trainIdx = trainIdx[:1000]
    valIdx = valIdx[:200]

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
        batchSize=256,
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



    model = LinearOperator([2+128*2, 200, 100, 50, 20, 1])
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 1000


    # Training loop
    log = {'train_loss': [] , 'val_loss': []}
    for epoch in range(num_epochs):
        running_loss = []
        for data in TrainLoader:
            # Get the input data and target outputs
            U, r, z, kappa = data
            U_m = U.reshape(U.shape[0], -1)
            r_real, r_imag = r[0], r[1]
            z_real, z_imag = z[0], z[1]

            z = torch.complex(z_real, z_imag)
            r = torch.complex(r_real, r_imag)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            #     with record_function("model_inference"):
            z_pred = model(r, U_m)
            
            #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            

            loss_real = criterion(z.real, z_pred.real)
            loss_imag = criterion(z.imag, z_pred.imag)
            loss = loss_real + loss_imag

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
        
        for data_val in ValLoader:
            U_val, r_val, z_val, kappa = data_val
            U_m_val = U_val.reshape(U_val.shape[0], -1)
            z_val = torch.complex(z_val[0], z_val[1])
            r_val = torch.complex(r_val[0], r_val[1])

            z_val_pred = model(r_val, U_m_val)
            loss_val_real = criterion(z_val.real, z_val_pred.real)
            loss_val_imag = criterion(z_val.imag, z_val_pred.imag)
            loss_val = loss_val_real + loss_val_imag

        train_loss = np.mean(running_loss)
        val_loss = loss_val.item()
        log['train_loss'].append(train_loss)
        log['val_loss'].append(val_loss)
        print(f'Epoch: {epoch + 1}, Loss: {train_loss:.3f} and {val_loss:.3f}')

    with open('../../DiracNPC/log/SPDkernel_integral.pkl', 'wb') as f:
        pickle.dump(log, f)
    print('Finished Training')