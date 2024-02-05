from typing import Any
import jax
import numpy as np
import optax
import jax.numpy as jnp
import wandb
from ..model.linearOpt import linearConvOpt
from flax.training import train_state
from ..utils.losses import PCG_loss, ILULoss
from functools import partial
from tqdm import tqdm
import os

# from ..utils.dirac import DiracOperator


key = jax.random.PRNGKey(0)
optimizer = optax.adam(learning_rate=0.001)

# # params = model.init(key, dataComb)
# variables = model.init(jax.random.key(0), dataComb, train=False)
# params = variables['params']
# batch_stats = variables['batch_stats']

# load data


def random_b(key, shape):
    # Generate random values for the real and imaginary parts
    real_part = 1 - jax.random.uniform(key, shape)
    imag_part = 1 - jax.random.uniform(jax.random.split(key)[1], shape)
    # Combine the real and imaginary parts
    complex_array = real_part + 1j * imag_part
    return complex_array


def mergRealImag(v):
    """
    The input v has shape (B, X, T, 4)
    the last two dims are real and imaginary part

    return shaep (B, X, T, 2), with complex entries
    that can be used in CG
    """
    v = v[..., :2] + 1j * v[..., -2:]
    return v
    pass


class TrainState(train_state.TrainState):
    batch_stats: Any


def init_train_state(model, random_key, shape, learning_rate) -> train_state.TrainState:
    # Initialize the Model
    variables = model.init(random_key, jnp.ones(shape), train=True)
    # Create the optimizer
    # optimizer = optax.adam(learning_rate)
    optimizer = optax.chain(
        optax.clip(1.0),
        optax.adamw(learning_rate=learning_rate),
    )
    # Create a State
    return TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
    )


@jax.jit
def average_gradient_norm(grads):
    grads = jax.tree_util.tree_leaves(grads)
    avg_grad_norm = jnp.mean(jnp.array([jnp.linalg.norm(g) for g in grads]))
    return avg_grad_norm


def train_step(state: train_state.TrainState, batch: jnp.ndarray, diracOpt, model, key):
    if os.environ.get("DATASET") == "ILU":
        in_mat, M, kappa = batch
    else:
        in_mat, kappa = batch
    kappa = float(kappa[0])

    # diracOpt = partial(diracOpt, U1=in_mat, kappa=0.276)

    _, key = jax.random.split(key)

    b = random_b(key, shape=in_mat.shape)

    U1_field = jnp.transpose(jnp.exp(1j * in_mat), axes=(0, 3, 1, 2))

    if os.environ.get("DATASET") == "ILU":
        gradient_fn = jax.value_and_grad(ILULoss, has_aux=True)
        (loss, updates), grads = jax.jit(gradient_fn, static_argnums=[3, 7])(
            state.params,
            batch_stats=state.batch_stats,
            x=b,
            model=model,
            in_mat=in_mat,
            m_mat=M,
            kappa=kappa,
            train=True,
        )
    else:
        gradient_fn = jax.value_and_grad(PCG_loss, has_aux=True)
        (loss, updates), grads = jax.jit(gradient_fn, static_argnums=[2, 7, 8, 9])(
            state.params,
            batch_stats=state.batch_stats,
            model=model,
            U1=U1_field,
            b=b,
            in_mat=in_mat,
            kappa=kappa,
            steps=20,
            operator=diracOpt,
            train=True,
        )

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])

    return state, loss, key, grads


def eval_step(state, batch, diracOpt, model, key):
    if os.environ.get("DATASET") == "ILU":
        in_mat, M, kappa = batch
    else:
        in_mat, kappa = batch

    kappa = float(kappa[0])
    # diracOpt = partial(diracOpt, U1=in_mat, kappa=0.276)

    _, key = jax.random.split(key)

    b = random_b(key, shape=in_mat.shape)
    U1_field = jnp.transpose(jnp.exp(1j * in_mat), axes=(0, 3, 1, 2))

    if os.environ.get("DATASET") == "ILU":
        loss, updates = jax.jit(ILULoss, static_argnums=[3, 7])(
            state.params,
            batch_stats=state.batch_stats,
            x=b,
            model=model,
            in_mat=in_mat,
            m_mat=M,
            kappa=kappa,
            train=False,
        )
    else:
        loss, updates = jax.jit(PCG_loss, static_argnums=[2, 7, 8, 9])(
            state.params,
            batch_stats=state.batch_stats,
            model=model,
            U1=U1_field,
            b=b,
            in_mat=in_mat,
            kappa=kappa,
            steps=20,
            operator=diracOpt,
            train=False,
        )
    return loss, key


def train_val(
    trainLoader, valLoader, state, epochs, diracOpt, model, verbose, log=False
):
    trainKey = jax.random.PRNGKey(0)
    valKey = jax.random.PRNGKey(1)
    for ep in range(epochs):
        # training
        trainBatchLoss = []
        gradients = []
        for train_batch in tqdm(trainLoader):
            if os.environ["DATASET"] == "ILU":
                batch = [
                    train_batch[0].numpy(),
                    train_batch[1].numpy(),
                    train_batch[2].numpy(),
                ]
            else:
                batch = [train_batch[0].numpy(), train_batch[1].numpy()]
            state, loss, trainKey, grads = train_step(
                state=state, batch=batch, diracOpt=diracOpt, model=model, key=trainKey
            )
            trainBatchLoss.append(loss)
            gradients.append(average_gradient_norm(grads))
            # break # just for prelim results
        valBatchLoss = []
        for val_batch in valLoader:
            if os.environ["DATASET"] == "ILU":
                Vbatch = [
                    val_batch[0].numpy(),
                    val_batch[1].numpy(),
                    val_batch[2].numpy(),
                ]
            else:
                Vbatch = [val_batch[0].numpy(), val_batch[1].numpy()]
            vLoss, valKey = eval_step(state, Vbatch, diracOpt, model, valKey)
            valBatchLoss.append(vLoss)
            # break  # only eval one batch
        if verbose == True:
            print(
                "Epoch {}, grads norm {:.4f} train loss {:.4f} validation loss {:.4f}".format(
                    ep + 1,
                    np.mean(gradients),
                    np.mean(trainBatchLoss),
                    np.mean(valBatchLoss),
                )
            )
        if log:
            wandb.log(
                {
                    "trainLoss": np.mean(trainBatchLoss),
                    "valLoss": np.mean(valBatchLoss),
                    "grads": np.mean(gradients),
                }
            )
    return state
