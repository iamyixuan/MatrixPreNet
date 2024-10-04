import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from deephyper.evaluator import RunningJob, profile
from precondCNN import (DataLoader, PrecondCNN, condition_number_loss,
                        inversion_loss


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


def condition_number(model, x, DD):
    # take the lower triangular part of the matrix
    L = jnp.tril(jax.vmap(model)(x).squeeze())
    L = model.scale * L + jnp.eye(L.shape[-1])
    LL_t = jnp.matmul(L, L.conj().transpose((0, 2, 1)))
    precond_sys = jnp.matmul(LL_t, DD)
    cond_number = jnp.linalg.cond(precond_sys)
    return jnp.mean(cond_number)


def train_hpo(
    model: PrecondCNN,
    trainloader: DataLoader,
    valloader: DataLoader,
    optim: optax.GradientTransformation,
    loss_name: str = "conditionNumber",
    configs: dict = None,
):
    if loss_name == "conditionNumber":
        loss_fn = condition_number_loss
    elif loss_name == "inversion":
        loss_fn = inversion_loss
    else:
        raise NotImplementedError

    loss_fn = eqx.filter_jit(loss_fn)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def update_step(model, x, DD, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, DD)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def val_step(model, x, DD):
        return loss_fn(model, x, DD)

    train_losses = []
    val_losses = []
    val_condNumbers = []
    for _ in range(200):
        running_loss = 0.0
        for i, (x, DD) in enumerate(trainloader.load_data()):
            model, opt_state, loss = update_step(model, x, DD, opt_state)
            running_loss += loss

        for x_val, DD_val in valloader.load_data():
            val_loss = val_step(model, x_val, DD_val)

        train_losses.append(float(running_loss / (i + 1)))
        val_losses.append(float(val_loss))
        val_condNumbers.append(float(condition_number(model, x_val, DD_val)))
        # print(
        #     "Train Loss:",
        #     train_losses[-1],
        #     "Val Loss:",
        #     val_losses[-1],
        #     "Val Cond Number:",
        #     val_condNumbers[-1],
        # )

    return train_losses, val_losses, val_condNumbers


@profile
def run(job: RunningJob):
    configs = job.parameters.copy()
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    activation = get_activation(configs["activation"])

    model = PrecondCNN(
        inch=2,
        outch=1,
        activation=activation,
        kernel_size=configs["kernel_size"],
        n_layers_gauge=configs["n_layers_gauge"],
        n_layers_precond=configs["n_layers_precond"],
        hidden_dim=configs["hidden_dim"],
        key=subkey,
    )

    data_path = "../data/U1_DD_matrices.pt"

    trainloader = DataLoader(
        data_path, mode="train", batch_size=configs["batch_size"]
    )
    valloader = DataLoader(data_path, mode="val")

    try:
        train_loss, val_loss, scores = train_hpo(
            model,
            trainloader,
            valloader,
            optax.adam(configs["lr"]),
            configs["loss"],
            configs,
        )

        metadata = {
            "train_losses": train_loss,
            "val_losses": val_loss,
        }

        final_objective = -scores[-1]  # maximize neg condition number
    except Exception as e:
        print(e)
        final_objective = "F"
        metadata = {}

    return {"objective": final_objective, "metadata": metadata}


if __name__ == "__main__":
    from deephyper.evaluator import Evaluator
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO

    problem = HpProblem()
    problem.add_hyperparameter(
        (16, 128),
        "hidden_dim",
        default_value=64,
    )  # hidden dimension
    problem.add_hyperparameter(
        (1, 10),
        "n_layers_gauge",
        default_value=3,
    )  # number of layers
    problem.add_hyperparameter(
        (1, 10),
        "n_layers_precond",
        default_value=3,
    )  # number of layers
    problem.add_hyperparameter(
        (1e-4, 1e-1, "log-uniform"),
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
        (1, 9),
        "kernel_size",
        default_value=3,
    )  # kernel size
    problem.add_hyperparameter(
        ["conditionNumber", "inversion"],
        "loss",
        default_value="conditionNumber",
    )  # kernel size
    # problem.add_hyperparameter(
    #     (100, 1000),
    #     "num_epochs",
    #     default_value=200,
    # )  # number of epochs

    evaluator = Evaluator.create(run)
    search = CBO(
        problem,
        evaluator,
        initial_points=[problem.default_configuration],
        verbose=1,
    )
    results = search.search(max_evals=100)
