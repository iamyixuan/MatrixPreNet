import os

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from deephyper.evaluator import RunningJob, profile
from precondFNN_U_tilde import (DataLoader, PrecondFNN, U1DDDataset,
                                condition_number_loss_tilde)
from src.utils.losses import inverse_loss


def get_activation(activation):
    if activation == "relu":
        return jax.nn.relu
    elif activation == "swish":
        return jax.nn.swish
    elif activation == "sigmoid":
        return jax.nn.sigmoid
    elif activation == "softplus":
        return jax.nn.softplus
    elif activation == "prelu":
        return eqx.nn.PReLU()
    else:
        raise ValueError(f"Activation {activation} not supported")


def train(
    model: PrecondFNN,
    trainloader: DataLoader,
    valloader: DataLoader,
    optim: optax.GradientTransformation,
    loss_name: str = "inverse",
    configs: dict = None,
):
    print(f"Training with {loss_name} loss")
    if loss_name == "conditionNumber":
        loss_fn = condition_number_loss_tilde
    elif loss_name == "inverse":
        loss_fn = inverse_loss
    else:
        raise NotImplementedError

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def update_step(model, inputs, opt_state):
        model = eqx.nn.inference_mode(model, value=False)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, inputs)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def val_step(model, inputs):
        model = eqx.nn.inference_mode(model)
        return loss_fn(model, inputs)


    train_losses = []
    val_losses = []
    scale = []
    for _ in range(configs["num_epochs"]):
        running_loss = 0.0
        for i, (U1, DD, tmask) in enumerate(trainloader):
            key = jax.random.PRNGKey(i)
            k1, k2 = jax.random.split(key)
            U1 = jnp.asarray(U1)
            DD = jnp.asarray(DD)
            tmask = jnp.nonzero(jnp.array(tmask))
            inputs = (U1, DD, tmask, k1)
            model, opt_state, loss = update_step(model, inputs, opt_state)
            running_loss += loss

        for U1_val, DD_val, vmask in valloader:
            U1_val = jnp.asarray(U1_val)
            DD_val = jnp.asarray(DD_val)

            vmask = jnp.nonzero(jnp.array(vmask))
            inputs_val = (U1_val, DD_val, vmask, k2)
            val_loss = val_step(model, inputs_val)

        train_losses.append(running_loss.item() / (i + 1))
        val_losses.append(val_loss.item())
        # scale.append(model.scale.item())

        print(f"Train loss: {train_losses[-1]}, Val loss: {val_losses[-1]}")
    return train_losses, val_losses, scale


def train_stage(configs):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    model = PrecondFNN(
        key=subkey,
        in_dim=128,
        out_dim=128,
        dropout_rate=configs["dropout_rate"],
        activation=get_activation(configs["activation"]),
        layer_sizes=[configs["hidden_dims"]] * configs["num_layers"],
    )

    train_config = {
        "num_epochs": 300,
        "batch_size": configs["batch_size"],
        "optim": optax.adam(configs["lr"]),
    }

    data_path = "../data/U1_DD_matrices.pt"
    trainset = U1DDDataset(data_path, mode="train")
    valset = U1DDDataset(data_path, mode="val")

    trainloader = DataLoader(trainset, batch_size=train_config["batch_size"])
    valloader = DataLoader(valset, batch_size=valset.__len__())

    train_losses, val_losses, scales = train(
        model,
        trainloader,
        valloader,
        train_config["optim"],
        "inverse",
        train_config,
    )
    return train_losses, val_losses, scales


@profile
def run(job: RunningJob):
    configs = job.parameters.copy()
    # try:
    train_loss, val_loss, scales = train_stage(configs)

    metadata = {
        "train_losses": train_loss,
        "val_losses": val_loss,
        # "scales": scales,
    }

    final_objective = -val_loss[-1]  # maximize neg condition number
    # except Exception as e:
    #     print(e)
    #     final_objective = "F"
    #     metadata = {}

    return {"objective": final_objective, "metadata": metadata}


if __name__ == "__main__":
    from deephyper.evaluator import Evaluator
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO

    problem = HpProblem()
    problem.add_hyperparameter(
        (16, 128),
        "hidden_dims",
        default_value=64,
    )  # hidden dimension
    problem.add_hyperparameter(
        (1, 20),
        "num_layers",
        default_value=3,
    )  # number of layers
    problem.add_hyperparameter(
        (1e-5, 1e-1, "log-uniform"),
        "lr",
        default_value=1e-3,
    )  # learning rate
    problem.add_hyperparameter(
        ["relu", "swish", "sigmoid", "softplus", "prelu"],
        "activation",
        default_value="relu",
    )  # activation function
    problem.add_hyperparameter(
        (1, 256),
        "batch_size",
        default_value=32,
    )  # batch size
    problem.add_hyperparameter(
        (0.0, 0.5),
        "dropout_rate",
        default_value=0.0,
    )  # dropout rate

    evaluator = Evaluator.create(run)
    # search = CBO(
    #     problem,
    #     evaluator,
    #     initial_points=[problem.default_configuration],
    #     verbose=1,
    # )
    log_dir = "hpo_precondFNN"
    os.makedirs(log_dir, exist_ok=True)
    search = CBO(
        problem,
        evaluator,
        log_dir=log_dir,
        initial_points=[problem.default_configuration],
        acq_optimizer="mixedga",
        acq_optimizer_freq=1,
        kappa=5.0,
        scheduler={
            "type": "periodic-exp-decay",
            "period": 50,
            "kappa_final": 0.0001,
        },
        objective_scaler="identity",
    )
    results = search.search(max_evals=500)
