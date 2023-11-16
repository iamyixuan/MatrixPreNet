from typing import Any
import jax
import numpy as np
import optax
import jax.numpy as jnp
import wandb
from flax.training import train_state
from ..utils.losses import PCG_loss
from functools import partial
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
    optimizer = optax.adam(learning_rate)
    # Create a State
    return TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
    )


#@jax.jit
def train_step(
    state: train_state.TrainState, 
    batch: jnp.ndarray,  
    diracOpt, 
    model,
    key
):
    in_mat, kappa = batch
    kappa = float(kappa[0])

    # diracOpt = partial(diracOpt, U1=in_mat, kappa=0.276)

    _, key = jax.random.split(key)

    b = random_b(key, shape=in_mat.shape)

    U1_field = jnp.transpose(in_mat, axes=(0, 3, 1, 2))
    lossFunc = partial(
        PCG_loss, U1=U1_field, b=b, kappa=kappa, steps=1000, operator=diracOpt
    )

    def NNopt(x):
        y_precond, _ = model.apply(
            {"params": state.params, "batch_stats": state.batch_stats}, x, mutable=['batch_stats'], train=True
        )
        return y_precond
    
    # x = random_b(key, b.shape)
    # print(x.shape)
    # y = NNopt(x)
    # print(y.shape)
    # assert False

    def loss_fn(params):
        pred, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            in_mat,
            train=True,
            mutable=["batch_stats"],
        )
        # loss = jnp.mean((pred - out_mat)**2)
        loss = lossFunc(NNopt)
        return loss, updates

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updates), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    return state, loss, key


#@jax.jit
def eval_step(state, batch,  diracOpt, model, key):
    in_mat, kappa = batch
    kappa = float(kappa[0])
    # diracOpt = partial(diracOpt, U1=in_mat, kappa=0.276)

    _, key = jax.random.split(key)

    b = random_b(key, shape=in_mat.shape)
    U1_field = jnp.transpose(in_mat, axes=(0, 3, 1, 2))

    def NNopt(x):
        y_precond = model.apply(
            {"params": state.params, "batch_stats": state.batch_stats}, x, mutable=False, train=False
        )
        return y_precond

    lossFunc = partial(
        PCG_loss, U1=U1_field, b=b, kappa=kappa, steps=1000, operator=diracOpt
    )
    loss = lossFunc(NNopt)

    # pred = state.apply_fn(
    #     {"params": state.params, "batch_stats": state.batch_stats},
    #     in_mat,
    #     train=False,
    #     mutable=False,
    # )
    return loss, key


def train_val(trainLoader, valLoader, state, epochs, diracOpt, model, verbose, log=False):
    trainKey = jax.random.PRNGKey(0)
    valKey = jax.random.PRNGKey(1)
    for ep in range(epochs):
        # training
        trainBatchLoss = []
        for train_batch in trainLoader:
            batch = [train_batch[0].numpy(), train_batch[1].numpy()]
            state, loss, trainKey = train_step(state=state, batch=batch, diracOpt=diracOpt, model=model, key=trainKey)
            trainBatchLoss.append(loss)
        valBatchLoss = []
        for val_batch in valLoader:
            Vbatch = [val_batch[0].numpy(), val_batch[1].numpy()]
            vLoss, valKey = eval_step(state, Vbatch, diracOpt, model, valKey)
            valBatchLoss.append(vLoss)
        if verbose == True:
            print(
                "Epoch {}, train loss {:.4f} validation loss {:.4f}".format(
                    ep + 1, np.mean(trainBatchLoss), np.mean(valBatchLoss)
                )
            )
        if log:
            wandb.log(
                {"trainLoss": np.mean(trainBatchLoss), "valLoss": np.mean(valBatchLoss)}
            )
    return state
