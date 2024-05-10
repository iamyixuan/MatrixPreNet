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

    U1 = torch.from_numpy(np.exp(1j * data))[
        :500
    ]  # use small subset for prototyping

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
    model = (
        NeuralPreconditioner(
            num_basis, DDOpt=DDOpt_torch, hidden_layers=hidden_layers
        )
        .to(device)
        .double()
    )

    loss_class = getLoss(trainConfig["loss"])
    loss_fn = loss_class(DDOpt_torch)
    loss_ortho = getLoss("BasisOrthoLoss")()

    # start of the training loop
    logger = Logger()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    NUM_EP = trainConfig["epochs"]

    batch_size = trainConfig["batch_size"]
    numIter = U1_train.shape[0] // batch_size

    for ep in tqdm(range(NUM_EP)):
        model.train()
        run_train_obj_loss = []
        run_train_basis_loss = []
        best_obj_loss = np.inf
        for x in TrainLoader:
            x = x.to(device)
            model.zero_grad()

            out = model(x)
            loss_obj = loss_fn(out, x)
            loss_basis = loss_ortho(model.basis_real, model.basis_imag)
            trainBatchLoss = loss_fn(out, x) + loss_ortho(
                model.basis_real, model.basis_imag
            )
            with torch.autograd.set_detect_anomaly(True):
                trainBatchLoss.backward(retain_graph=False)
            optimizer.step()

            run_train_obj_loss.append(loss_obj.detach().cpu().item())
            run_train_basis_loss.append(loss_basis.detach().cpu().item())
        for x_val in ValLoader:
            model.eval()
            x_val = x_val.to(device)
            out_val = model(x_val)
            loss_obj_val = loss_fn(out_val, x_val)
            loss_basis_val = loss_ortho(model.basis_real, model.basis_imag)

        logger.record("train_obj_loss", np.mean(run_train_obj_loss))
        logger.record("train_basis_loss", np.mean(run_train_basis_loss))
        logger.record("val_obj_loss", loss_obj_val.cpu().item())
        logger.record("val_basis_loss", loss_basis_val.cpu().item())

        if ep % 10 == 0:
            logger.record(
                "learned_vector_real", model.basis_real.detach().cpu().numpy()
            )
            logger.record(
                "learned_vector_imag", model.basis_imag.detach().cpu().numpy()
            )
            # save the model
        if best_obj_loss > loss_obj_val.item():
            torch.save(
                model.state_dict(),
                trainConfig["log_path"]
                + f"/best_{trainConfig['model_name']}_model.pth",
            )

        print(
            f"Epoch {ep + 1}, Train Loss obj|basis: {np.mean(run_train_obj_loss):.4f}|{np.mean(run_train_basis_loss):.4f}, Val Loss obj|basis: {loss_obj_val.item():.4f}|{loss_basis_val.item():.4f}"
        )

    with open(
        trainConfig["log_path"] + f"/{now}_{trainConfig['log_name']}.pkl", "wb"
    ) as f:
        pickle.dump(logger.logger, f)
