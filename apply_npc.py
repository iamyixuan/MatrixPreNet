import pickle
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from NeuralPC.model.neural_preconditioner import NeuralPreconditioner
from NeuralPC.utils.data import U1Data, split_data_idx
from NeuralPC.utils.dirac import DDOpt_torch
from NeuralPC.utils.logger import Logger
from NeuralPC.utils.losses_torch import getLoss

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("./config.yaml") as f:
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
    U1_mat = torch.from_numpy(U1_mat).cdouble()
    U1 = torch.from_numpy(np.exp(1j * data))
    train_idx, val_idx = split_data_idx(U1.shape[0])


    data = U1Data(U1[train_idx])

    data_load = DataLoader(data, batch_size=32)

    num_basis = modelConfig["num_basis"]

    hidden_layers = (
        [2 * 128 * num_basis]
        + modelConfig["hidden_layers"]
        + [2 * modelConfig["out_size"]]
    )
    model = (
        NeuralPreconditioner(num_basis, DDOpt=DDOpt_torch, hidden_layers=hidden_layers)
        .to(device)
        .double()
    )
    # load saved model
    model.load_state_dict(torch.load("./logs/2024-04-18-16_ep20_LL_cg_loss.pth"))

    loss_class = getLoss(trainConfig["loss"])
    loss_fn = loss_class(DDOpt_torch, verbose=True)
    loss_ortho = getLoss("BasisOrthoLoss")()

    # start of the training loop
    model.eval()
    for x in data_load:
        x = x.to(device)
        out = model(x)
        loss_obj, info = loss_fn(out, x)
        loss_basis_val = loss_ortho(model.basis_real, model.basis_imag)
        print(loss_obj)
        print(loss_basis_val)
        break

    # with open("./logs/npc_cg_solve.pkl", "wb") as f:
    #     pickle.dump(info, f)
