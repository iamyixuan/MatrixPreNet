import pickle
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from NeuralPC.model.neural_preconditioner import NeuralPreconditioner
from NeuralPC.utils.data import U1Data, split_data_idx
from NeuralPC.utils.dirac import DDOpt_torch
from NeuralPC.utils.losses_torch import GetBatchMatrix, getLoss

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("./config.yaml") as f:
        config = yaml.safe_load(f)

    dataConfig = config["data"]
    modelConfig = config["model"]
    trainConfig = config["train"]

    now = datetime.now().strftime("%Y-%m-%d-%H")
    model_name = "best_LL_T_left_pc_true_condNum_loss_penalized_5_model.pth"

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

    data = U1Data(U1)

    data_load = DataLoader(data, batch_size=data.__len__())

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
    # load saved model
    model.load_state_dict(
        torch.load(
            f"./logs/{model_name}"
        )
    )

    loss_class = getLoss("CG_loss")
    loss_fn = loss_class(DDOpt_torch, maxiter=500, verbose=True)
    loss_ortho = getLoss("BasisOrthoLoss")()

    spec_loss = getLoss("SpectrumLoss")(DDOpt_torch)

    # start of the training loop
    model.eval()
    for x in data_load:
        x = x.to(device)
        out = model(x)
        # loss_obj, info = loss_fn(out, x)
        # loss_basis_val = loss_ortho(model.basis_real, model.basis_imag)
        loss = spec_loss(out, x)
        print("loss:", loss.item())
        org_e = spec_loss.org_spectrum(out, x)
        pc_e = spec_loss.pc_spectrum(out, x)

        # also get the true SVD of the original system and the preconditioned system
        # TODO: 1. matrix getter; 2. perform SVD; 3. get the spectrum
        # I can implement this as additional method to the loss class
        org_true_e = spec_loss.org_spectrum(out, x, True)
        pc_true_e = spec_loss.pc_spectrum(out, x, True)
        lower_L_e, L = spec_loss.lower_spectrum(out, x, True)

    spectra = {
        "org": org_e,
        "pc": pc_e,
        "org_true": org_true_e,
        "pc_true": pc_true_e,
        "lower_L": lower_L_e,
        "L": L,
    }

    with open(f"./logs/{model_name[:-4]}-test-spectrum.pkl", "wb") as f:
        pickle.dump(spectra, f)
