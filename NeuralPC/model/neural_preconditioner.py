import torch
import torch.nn as nn


class NeuralPreconditioner(nn.Module):
    def __init__(self, basis_size, DDOpt, hidden_layers):
        # the last number of the hidden layer must be 128 * 128 for simple linear map
        super().__init__()

        self.DDOpt = DDOpt
        # Create basis of trainable vectors
        self.basis_real = nn.Parameter(torch.randn(basis_size, 8, 8, 2))
        self.basis_imag = nn.Parameter(torch.randn(basis_size, 8, 8, 2))

        self.layers = nn.ModuleList()
        # create non-linear layers to output preconditioners

        for k in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[k], hidden_layers[k + 1]))
            self.layers.append(nn.PReLU())
            self.layers.append(nn.BatchNorm1d(hidden_layers[k + 1]))

    def forward(self, U1):
        basis = torch.complex(self.basis_real, self.basis_imag)
        # x is a random vector that is used to create the
        output_matrix = []
        for i in range(basis.shape[0]):
            output_matrix.append(
                self.DDOpt(basis[i : i + 1], U1, kappa=0.276).reshape(U1.shape[0], -1)
            )  # each instance should be of shape B, 128

        # the formed matrix approximation should be of shape B, num_basis, 128
        x = torch.stack(output_matrix, dim=1)

        # separate the real and imaginary parts
        x_real, x_imag = x.real, x.imag

        # create additional dim for imaginary parts
        x = torch.cat([x_real, x_imag], dim=1).reshape(x.shape[0], -1)

        for layer in self.layers:
            x = layer(x)

        # the output would be the weights for the linear combination of the resulting vectors
        # it should have the shape of 2*128 * 128, then reshape to B, 128, 128, 2 and then combine the channels to complex entries

        x = x.reshape(x.shape[0], 128, 128, 2)
        x_real = x[..., 0]
        x_imag = x[..., 1]

        return torch.complex(x_real, x_imag)


class MatrixConditionNumber(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, matrix):
        U, S, V = torch.linalg.svd(matrix)  # S contains the singular values

        max_s = S[0]
        min_s = S[-1]

        return max_s / min_s


class ConditionNumberLoss(nn.Module):
    def __init__(self, DDOpt):
        super().__init__()
        self.DDOpt = DDOpt

    def forward(self, net_out, U1):
        # calculate M_inv D D_dag v
        shape = (net_out.shape[0], 8, 8, 2)
        v_real = torch.rand(shape)
        v_imag = torch.rand(shape)

        # v of shape B, 8, 8, 2
        v = torch.complex(v_real, v_imag)
        condition_num = self.power_method(U1, net_out, v)
        return torch.norm(condition_num)

    def power_method(self, U1, net_out, v, num_iter=100):

        lambda_max = self.get_largest_eigen(U1, net_out, v, num_iter)
        lambda_min = self.get_smallest_eigen(U1, net_out, v, num_iter, lambda_max)

        max_abs = torch.abs(lambda_max)
        min_abs = torch.abs(lambda_min)
        return max_abs / min_abs

    def get_largest_eigen(self, U1, net_out, v, num_iter):
        for i in range(num_iter):
            v_k = self.matrix_vec_prod(U1, net_out, v)
            v = v_k / torch.norm(v_k)

        Av = self.matrix_vec_prod(U1, net_out, v)

        eigenvalue = self.b_dot(Av, v) / self.b_dot(v, v)
        return eigenvalue

    def get_smallest_eigen(self, U1, net_out, v, num_iter, max_eigen):
        max_eigen = max_eigen.view(-1, 1)

        for i in range(num_iter):
            v_k = self.spectral_shift(U1, net_out, v, max_eigen)
            v = v_k / torch.norm(v_k)

        Av = self.spectral_shift(U1, net_out, v, max_eigen)
        eigenvalue = self.b_dot(Av, v) / self.b_dot(v, v)

        return eigenvalue + max_eigen.view(
            -1,
        )

    def matrix_vec_prod(self, U1, net_out, v):
        v = v.reshape(v.shape[0], 8, 8, 2)
        v_temp = self.DDOpt(v, U1, kappa=0.276)
        return self.M(net_out, v_temp)

    def spectral_shift(self, U1, net_out, v, max_eigen):
        return self.matrix_vec_prod(U1, net_out, v) - max_eigen * v.reshape(
            v.shape[0], -1
        )

    def M(self, net_out, v):
        v = v.reshape(v.shape[0], -1)
        M_v = torch.einsum("bij, bj -> bi", net_out, v)
        return M_v

    def b_dot(self, v, w):
        return torch.einsum("bi, bi -> b", v, w.conj())


class DDApprox(nn.Module):
    def __init__(self, basis_size, DDOpt):
        # the last number of the hidden layer must be 128 * 128 for simple linear map
        super().__init__()

        self.DDOpt = DDOpt
        # Create basis of trainable vectors
        self.basis_real = nn.Parameter(torch.randn(basis_size, 8, 8, 2))
        self.basis_imag = nn.Parameter(torch.randn(basis_size, 8, 8, 2))

    def forward(self, U1):
        basis = torch.complex(self.basis_real, self.basis_imag)
        # x is a random vector that is used to create the
        output_matrix = []
        for i in range(basis.shape[0]):
            output_matrix.append(
                self.DDOpt(basis[i : i + 1], U1, kappa=0.276).reshape(U1.shape[0], -1)
            )  # each instance should be of shape B, 128

        # the formed matrix approximation should be of shape B, num_basis, 128
        x = torch.stack(output_matrix, dim=1)

        return x


# calculate the matrix form of the Dirac operator
# implemenent GetMatrix in pytorch
class GetMatrix:
    def __init__(self, DDOpt, kappa=0.276) -> None:
        self.kappa = kappa
        self.DDOpt = DDOpt

    def getMatrix(self, U1):
        """
        Get the matrix of the original system
        """
        n = 128
        A = torch.zeros((n, n)).cfloat()
        for i in range(n):
            A = self._getEntry(A, U1, i)
        return A

    def _getEntry(self, A, U1, i):
        n = A.shape[0]
        e_i = torch.zeros(n).float()
        e_i[i] = 1
        A[:, i] = self.DDOpt(e_i.reshape(1, 8, 8, 2), U1=U1, kappa=self.kappa).ravel()
        return A


class ComplexMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.norm(x - y)


class ComplexMAEMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.norm(x - y, p=1) + torch.norm(x - y, p=2)


def getBatchMatrix(DDOpt, U1):
    getter = GetMatrix(DDOpt)
    for i in range(U1.shape[0]):
        A = getter.getMatrix(U1[i : i + 1])
        if i == 0:
            batch = A.reshape(1, 128, 128)
        else:
            batch = np.concatenate([batch, A.reshape(1, 128, 128)], axis=0)
    return torch.from_numpy(batch).cfloat()


if __name__ == "__main__":
    import pickle
    from datetime import datetime

    import numpy as np
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from NeuralPC.utils.data import U1Data, split_data_idx
    from NeuralPC.utils.dirac import DDOpt_torch
    from NeuralPC.utils.logger import Logger

    now = datetime.now().strftime("%Y-%m-%d-%H")

    data = np.load(
        "/Users/yixuan.sun/Documents/projects/Preconditioners/datasets/Dirac/precond_data/config.l8-N1600-b2.0-k0.276-unquenched.x.npy"
    )
    U1_mat = np.load(
        "/Users/yixuan.sun/Documents/projects/Preconditioners/datasets/Dirac/precond_data/U1_mat.npy",
    )

    U1_mat = torch.from_numpy(U1_mat).cfloat()
    print(U1_mat[0])

    # data = data[:200]
    # expoential transform
    U1 = torch.from_numpy(np.exp(1j * data)).cfloat()

    train_idx, val_idx = split_data_idx(U1.shape[0])

    U1_train = U1[train_idx]
    U1_mat_train = U1_mat[train_idx]
    U1_val = U1[val_idx]
    U1_mat_val = U1_mat[val_idx]

    print("Data prepared")

    trainData = U1Data(U1_train)
    valData = U1Data(U1_val)

    TrainLoader = DataLoader(trainData, batch_size=256, shuffle=True)
    ValLoader = DataLoader(valData, batch_size=valData.__len__())

    num_basis = 128

    hidden_layers = [2 * 128 * num_basis] + [1024] * 5 + [2 * 128 * 128]
    # model = NeuralPreconditioner(
    #     num_basis, DDOpt=DDOpt_torch, hidden_layers=hidden_layers
    # )
    model = DDApprox(num_basis, DDOpt=DDOpt_torch)
    model.double()
    # loss_fn = ConditionNumberLoss(DDOpt=DDOpt_torch)
    loss_fn = ComplexMSE()

    # start of the training loop
    logger = Logger()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    NUM_EP = 1000

    batch_size = 256
    numIter = U1_train.shape[0] // batch_size
    for ep in tqdm(range(NUM_EP)):
        model.train()
        runTrainLoss = []
        # for x in TrainLoader:
        #     model.zero_grad()
        #
        #     out = model(x)
        #     mat = getBatchMatrix(DDOpt_torch, x)
        #     trainBatchLoss = loss_fn(out, mat)
        #     with torch.autograd.set_detect_anomaly(True):
        #         trainBatchLoss.backward(retain_graph=False)
        #     optimizer.step()
        #
        #     runTrainLoss.append(trainBatchLoss.detach().item())
        # for x_val in ValLoader:
        #     model.eval()
        #     out_val = model(x_val)
        #     mat_val = getBatchMatrix(DDOpt_torch, x_val)
        #     val_loss = loss_fn(out_val, mat_val)

        # new training
        for i in range(numIter):
            model.zero_grad()
            x = U1_train[i * batch_size : (i + 1) * batch_size]
            x_mat = U1_mat_train[i * batch_size : (i + 1) * batch_size]
            out = model(x)
            trainBatchLoss = loss_fn(out, x_mat)
            with torch.autograd.set_detect_anomaly(True):
                trainBatchLoss.backward(retain_graph=False)
            optimizer.step()

            runTrainLoss.append(trainBatchLoss.detach().item())

        model.eval()
        out_val = model(U1_val)
        mat_val = U1_mat_val
        val_loss = loss_fn(out_val, mat_val)

        logger.record("TrainLoss", np.mean(runTrainLoss))
        logger.record("ValLoss", val_loss.item())

        if ep % 10 == 0:
            logger.record("learned_vector_real", model.basis_real.detach().numpy())
            logger.record("learned_vector_imag", model.basis_imag.detach().numpy())

        print(
            f"Epoch {ep + 1}, Train Loss: {np.mean(runTrainLoss):.4f}, Val Loss: {val_loss.item():.4f}"
        )

    with open(f"../../logs/{now}_DDApprox_MAEtrainLog.pkl", "wb") as f:
        pickle.dump(logger.logger, f)
