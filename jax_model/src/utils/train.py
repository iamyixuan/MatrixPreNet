import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from torch.utils.data import DataLoader, Dataset

from .losses import (condition_number_loss, condition_number_loss_coo,
                     condition_number_loss_with_mask)

logger = logging.getLogger(__name__)


def get_optimizer(optimizer_config, learning_rate):
    if optimizer_config == "adam":
        # schedule = optax.warmup_cosine_decay_schedule(
        #     init_value=0,
        #     peak_value=learning_rate,
        #     warmup_steps=100,
        #     decay_steps=5000,
        #     end_value=1e-6,
        # )
        return optax.adam(learning_rate)
    elif optimizer_config == "sgd":
        return optax.sgd(learning_rate)
    else:
        raise NotImplementedError


def train(
    model: eqx.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    configs: dict = None,
):
    loss_name = configs["loss_name"]
    print(f"Training with {loss_name} loss")
    if loss_name == "ConditionNumberLoss":
        loss_fn = condition_number_loss
    else:
        raise NotImplementedError

    optim = get_optimizer(configs["optimizer"], configs["lr"])
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

    best_loss = jnp.inf
    for _ in range(configs["epochs"]):
        running_loss = 0.0
        for i, (x, DD) in enumerate(trainloader):
            x = jnp.array(x)
            DD = jnp.array(DD)
            model, opt_state, loss = update_step(model, x, DD, opt_state)
            running_loss += loss

        for x_val, DD_val in valloader:
            x_val = jnp.array(x_val)
            DD_val = jnp.array(DD_val)
            val_loss = val_step(model, x_val, DD_val)

        if val_loss < best_loss:
            best_loss = val_loss
            eqx.tree_serialise_leaves(
                f"./logs/{configs['logname']}/best_model.eqx", model
            )

        print(f"train Loss: {running_loss / (i + 1)}")
        print(f"scale: {model.alpha}")
        print(f"val Loss: {val_loss}")
        logger.info(
            f"train Loss: {running_loss / (i + 1)}, val Loss: {val_loss}, scale: {model.alpha}"
        )
        # print(f"Time taken: {time.time() - current_time}")
    return model


def test(
    model: eqx.Module,
    testloader: DataLoader,
    configs: dict = None,
):
    if configs.model_path is not None:
        model = eqx.tree_deserialise_leaves(configs.model_path, model)
        # inference mode
        inference_model = eqx.nn.inferece_mode(model)

        predicted = []
        for x_test, DD_test in testloader.load_data():
            x_test = jnp.array(x_test)
            DD_test = jnp.array(DD_test)
            pred = jax.vmap(inference_model)(x_test, DD_test)
            predicted.append(pred)
        predicted = jnp.concatenate(predicted, axis=0)
        return predicted
    else:
        raise NotImplementedError


def train_with_mask(
    model: eqx.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    configs: dict = None,
):
    loss_name = configs["loss_name"]
    print(f"Training with {loss_name} loss")
    if loss_name == "ConditionNumberLoss":
        loss_fn = condition_number_loss_with_mask
    else:
        raise NotImplementedError

    optim = get_optimizer(configs["optimizer"], configs["lr"])
    loss_fn = eqx.filter_jit(loss_fn)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def update_step(model, x, DD, mask, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, DD, mask)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def val_step(model, x, DD, mask):
        return loss_fn(model, x, DD, mask)

    best_loss = jnp.inf
    for _ in range(configs["epochs"]):
        running_loss = 0.0
        for i, (x, DD, mask) in enumerate(trainloader):
            x = jnp.array(x)
            x = x.reshape(x.shape[0], -1, 1)
            DD = jnp.array(DD)
            mask = jnp.nonzero(jnp.array(mask))
            model, opt_state, loss = update_step(model, x, DD, mask, opt_state)
            running_loss += loss

        for x_val, DD_val, val_mask in valloader:
            x_val = jnp.array(x_val)
            DD_val = jnp.array(DD_val)
            x_val = x_val.reshape(x_val.shape[0], -1, 1)
            val_mask = jnp.nonzero(jnp.array(val_mask))
            val_loss = val_step(model, x_val, DD_val, val_mask)

        if val_loss < best_loss:
            best_loss = val_loss
            eqx.tree_serialise_leaves(
                f"./logs/{configs['logname']}/best_model.eqx", model
            )

        print(f"train Loss: {running_loss / (i + 1)}")
        print(f"scale: {model.alpha}")
        print(f"val Loss: {val_loss}")
        logger.info(
            f"train Loss: {running_loss / (i + 1)}, val Loss: {val_loss}, scale: {model.alpha}"
        )
        # print(f"Time taken: {time.time() - current_time}")
    return model


def train_COO(
    model: eqx.Module,
    trainset: Dataset,
    valset: Dataset,
    configs: dict = None,
):
    loss_name = configs["loss_name"]
    print(f"Training with {loss_name} loss")
    if loss_name == "ConditionNumberLoss":
        loss_fn = condition_number_loss_coo
    else:
        raise NotImplementedError

    trainloader = DataLoader(
        trainset, batch_size=configs["batch_size"], shuffle=True
    )
    valloader = DataLoader(valset, batch_size=valset.__len__(), shuffle=False)
    adj_mat = trainset.adj_matrix

    optim = get_optimizer(configs["optimizer"], configs["lr"])
    # loss_fn = eqx.filter_jit(loss_fn)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # @eqx.filter_jit
    def update_step(model, x, edge, DD, adj_mat, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            model, x, edge, DD, adj_mat
        )
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # @eqx.filter_jit
    def val_step(model, x, edge, DD, adj_mat):
        return loss_fn(model, x, edge, DD, adj_mat)

    best_loss = jnp.inf
    for _ in range(configs["epochs"]):
        running_loss = 0.0
        for i, (x, edge, DD) in enumerate(trainloader):
            x = jnp.array(x)
            x = x.reshape(x.shape[0], -1, 1)
            edge = jnp.array(edge)
            DD = jnp.array(DD)

            model, opt_state, loss = update_step(
                model, x, edge, DD, adj_mat, opt_state
            )
            running_loss += loss

        for x_val, edge_val, DD_val in valloader:
            x_val = jnp.array(x_val)
            x_val = x_val.reshape(x_val.shape[0], -1, 1)
            edge_val = jnp.array(edge_val)
            DD_val = jnp.array(DD_val)
            val_loss = val_step(model, x_val, edge_val, DD_val, adj_mat)

        if val_loss < best_loss:
            best_loss = val_loss
            eqx.tree_serialise_leaves(
                f"./logs/{configs['logname']}/best_model.eqx", model
            )

        print(f"train Loss: {running_loss / (i + 1)}")
        print(f"scale: {model.alpha}")
        print(f"val Loss: {val_loss}")
        logger.info(
            f"train Loss: {running_loss / (i + 1)}, val Loss: {val_loss}, scale: {model.alpha}"
        )
        # print(f"Time taken: {time.time() - current_time}")
    return model
