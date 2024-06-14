import numpy as np
from tqdm import tqdm
import torch
from deephyper.evaluator import RunningJob, profile
from torch.optim.lr_scheduler import CosineAnnealingLR

from NeuralPC.model import FNN
from NeuralPC.utils.losses_torch import getLoss
from NeuralPC.utils.pair_data import get_dataset


def get_activation(activation):
    if activation == "relu":
        return torch.nn.ReLU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation == "softplus":
        return torch.nn.Softplus()
    elif activation == "prelu":
        return torch.nn.PReLU()
    else:
        raise ValueError(f"Activation {activation} not supported")


@profile
def run(job: RunningJob):
    config = job.parameters.copy()
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    in_dim = 1792
    out_dim = 960
    model = FNN(
        in_dim=in_dim,
        out_dim=out_dim,
        activation=get_activation(config["activation"]),
        layer_sizes=config["num_layers"] * [config["hidden_dim"]],
    )

    # load data for HPO
    data_dir = "./data/DD_mat_IC_L.pt"
    data_name = "DD_IC"
    dataset = get_dataset(data_name)(
        data_dir,
        config["batch_size"],
        shuffle=True,
        validation_split=0.2,
    )

    train_loader, val_loader = dataset.get_data_loader()

    # specify optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    loss_name = "MatConditionNumberLoss"
    loss_fn = getLoss(loss_name, mask=True)

    # training loop
    train_losses = []
    val_losses = []

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in tqdm(range(config["num_epochs"])):
        model.train()
        running_loss = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_fn(X, y_pred, scale=model.scale) # unsupervised loss
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        scheduler.step()
        train_losses.append(running_loss / len(train_loader))

        # validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, (X, y) in enumerate(val_loader):
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                val_loss += loss_fn(X, y_pred, scale=model.scale)
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

    metadata = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    final_loss = val_losses[-1].item()
    return {"objective": -final_loss, "metadata": metadata}


if __name__ == "__main__":
    from deephyper.evaluator import Evaluator
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO

    problem = HpProblem()
    problem.add_hyperparameter(
        (16, 1024),
        "hidden_dim",
        default_value=64,
    )  # hidden dimension
    problem.add_hyperparameter(
        (1, 10),
        "num_layers",
        default_value=3,
    )  # number of layers
    problem.add_hyperparameter(
        (1e-4, 1e-1, "log-uniform"),
        "lr",
        default_value=1e-3,
    )  # learning rate
    problem.add_hyperparameter(
        ["relu", "tanh", "sigmoid", "softplus", "prelu"],
        "activation",
        default_value="relu",
    )  # activation function
    problem.add_hyperparameter(
        (1, 256),
        "batch_size",
        default_value=32,
    )  # batch size
    problem.add_hyperparameter(
        (100, 1000),
        "num_epochs",
        default_value=200,
    )  # number of epochs

    evaluator = Evaluator.create(run)
    search = CBO(
        problem,
        evaluator,
        initial_points=[problem.default_configuration],
        verbose=1,
    )
    results = search.search(max_evals=20)
