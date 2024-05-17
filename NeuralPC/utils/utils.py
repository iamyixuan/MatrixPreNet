import numpy as np

from .losses_torch import getLoss


def get_trainer(trainer, **kwargs):

    if kwargs.get("loss_fn") is not None:
        loss_fn = getLoss(kwargs["loss_fn"])
    else:
        loss_fn = getLoss("MSELoss")

    if kwargs.get("optimizer_nm") is not None:
        optimizer = kwargs["optimizer_nm"]
        if optimizer == "Adam":
            from torch.optim import Adam

            optimizer = Adam
        elif optimizer == "SGD":
            from torch.optim import SGD

            optimizer = SGD
        elif optimizer == "RMSprop":
            from torch.optim import RMSprop

            optimizer = RMSprop
        elif optimizer == "Adadelta":
            from torch.optim import Adadelta

            optimizer = Adadelta
        else:
            raise NotImplementedError

    if kwargs.get("model_type") is not None:
        model_type = kwargs["model_type"]
        print(model_type)
        if model_type == "neural_pc":
            from .models.neural_pc import NeuralPC

            model = NeuralPC()
        elif model_type == "linear_inverse":
            from ..model.linear_inv import LinearInverse

            model = LinearInverse(10)  # number of layers
        elif model_type == "FNN":
            from ..model.FNN import FNN

            model_kwargs = {
                "in_dim": kwargs["in_dim"],
                "out_dim": kwargs["out_dim"],
            }

            model = FNN(**model_kwargs)
        elif model_type == "RNN":
            from ..model.models import RNN
            model_kwargs = {
                "in_dim": kwargs["in_dim"],
                "hidden_dim": kwargs["hidden_dim"],
                "out_dim": kwargs["out_dim"],
            }
            model = RNN(**model_kwargs)
        else:
            raise NotImplementedError

    if trainer == "supervised":
        from ..train import SupervisedTrainer

        print(optimizer)
        return SupervisedTrainer(model, optimizer, loss_fn, **kwargs)
    else:
        raise NotImplementedError


def split_idx(idx_len):
    rs = np.random.RandomState(0)
    idx = rs.permutation(idx_len)
    train_size = int(0.5 * idx_len)
    test_size = int(0.25 * idx_len)

    train_idx = idx[:train_size]
    val_idx = idx[train_size : train_size + test_size]
    test_idx = idx[-test_size:]
    return train_idx, val_idx, test_idx
