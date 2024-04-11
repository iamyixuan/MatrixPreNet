import torch
import torch.nn as nn

from NeuralPC.utils.losses_torch import get_loss


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
            if k < len(hidden_layers) - 2:
                self.layers.append(nn.PReLU())
                self.layers.append(nn.BatchNorm1d(hidden_layers[k + 1]))
            else:
                self.layers.append(nn.ReLU())

    def forward(self, U1):
        basis = torch.complex(self.basis_real, self.basis_imag)
        # x is a random vector that is used to create the
        output_matrix = []
        # approximating the matrix form of the Dirac operator
        # this ideally should produce the SVD of the matrix.
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
        x_real = x[..., : x.size(1) // 2]
        x_imag = x[..., x.size(1) // 2 :]

        # x = x.reshape(x.shape[0], 8, 8, 2, 2)  # to match the shape of U1
        # x_real = x[..., 0]
        # x_imag = x[..., 1]
        return torch.complex(x_real, x_imag)


if __name__ == "__main__":
    import pickle
    from datetime import datetime

    import numpy as np
    import yaml
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from NeuralPC.utils.data import U1Data, split_data_idx
    from NeuralPC.utils.dirac import DDOpt_torch
    from NeuralPC.utils.logger import Logger

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("../../config.yaml") as f:
        config = yaml.safe_load(f)

    dataConfig = config["data"]
    modelConfig = config["model"]
    trainConfig = config["train"]

    now = datetime.now().strftime("%Y-%m-%d-%H")

    U1_path = (
        dataConfig["path"]
        + "/precond_data/config.l8-N1600-b2.0-k0.276-unquenched.x.npy"
    )
    U1_mat_path = dataConfig["path"] + "precond_data/U1_mat.npy"

    data = np.load(U1_path)
    U1_mat = np.load(U1_mat_path)
    U1_mat = torch.from_numpy(U1_mat).cfloat()

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

    TrainLoader = DataLoader(
        trainData, batch_size=trainConfig["batch_size"], shuffle=True
    )
    ValLoader = DataLoader(valData, batch_size=valData.__len__())

    num_basis = modelConfig["num_basis"]

    hidden_layers = (
        [2 * 128 * num_basis]
        + modelConfig["hidden_layers"]
        + [2 * modelConfig["out_size"]]
    )
    model = NeuralPreconditioner(
        num_basis, DDOpt=DDOpt_torch, hidden_layers=hidden_layers
    ).to(device)

    # model = DDApprox(num_basis, DDOpt=DDOpt_torch)
    # model.double()
    # loss_fn = ConditionNumberLoss(DDOpt=DDOpt_torch)
    # loss_fn = ComplexMSE()
    loss_class = get_loss(trainConfig["loss"])
    loss_fn = loss_class(DDOpt_torch)
    loss_ortho = get_loss('BasisOrthoLoss')()

    # start of the training loop
    logger = Logger()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    NUM_EP = trainConfig["epochs"]

    batch_size = trainConfig["batch_size"]
    numIter = U1_train.shape[0] // batch_size

    for ep in tqdm(range(NUM_EP)):
        model.train()
        runTrainLoss = []
        for x in TrainLoader:
            x = x.to(device)
            model.zero_grad()

            out = model(x)
            trainBatchLoss = loss_fn(out, x) + loss_ortho(
                model.basis_real, model.basis_imag
            )
            with torch.autograd.set_detect_anomaly(True):
                trainBatchLoss.backward(retain_graph=False)
            optimizer.step()

            runTrainLoss.append(trainBatchLoss.detach().cpu().item())
        for x_val in ValLoader:
            model.eval()
            x_val = x_val.to(device)
            out_val = model(x_val)
            val_loss = loss_fn(out_val, x_val) + loss_ortho(
                model.basis_real, model.basis_imag
            )

        # new training
        # for i in range(numIter):
        #     model.zero_grad()
        #     x = U1_train[i * batch_size : (i + 1) * batch_size]
        #     x_mat = U1_mat_train[i * batch_size : (i + 1) * batch_size]
        #     out = model(x)
        #     trainBatchLoss = loss_fn(out, x_mat)
        #     with torch.autograd.set_detect_anomaly(True):
        #         trainBatchLoss.backward(retain_graph=False)
        #     optimizer.step()
        #
        #     runTrainLoss.append(trainBatchLoss.detach().item())

        # model.eval()
        # out_val = model(U1_val)
        # mat_val = U1_mat_val
        # val_loss = loss_fn(out_val, mat_val)

        logger.record("TrainLoss", np.mean(runTrainLoss))
        logger.record("ValLoss", val_loss.cpu().item())

        if ep % 10 == 0:
            logger.record(
                "learned_vector_real", model.basis_real.detach().cpu().numpy()
            )
            logger.record(
                "learned_vector_imag", model.basis_imag.detach().cpu().numpy()
            )

        print(
            f"Epoch {ep + 1}, Train Loss: {np.mean(runTrainLoss):.4f}, Val Loss: {val_loss.item():.4f}"
        )

    with open(
        trainConfig["log_path"] + f"/{now}_K-cond_TwoSide_basisOrtho.pkl", "wb"
    ) as f:
        pickle.dump(logger.logger, f)
