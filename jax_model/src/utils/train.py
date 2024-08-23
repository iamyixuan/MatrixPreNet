import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from .losses import condition_number_loss
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def get_optimizer(optimizer_config, learning_rate):
    if optimizer_config == "adam":
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
            print(x.shape, DD.shape)
            model, opt_state, loss = update_step(model, x, DD, opt_state)
            running_loss += loss

        for x_val, DD_val in valloader.load_data():
            x_val = jnp.array(x_val)
            DD_val = jnp.array(DD_val)
            val_loss = val_step(model, x_val, DD_val)

        if val_loss < best_loss:
            best_loss = val_loss
            eqx.tree_serialise_leaves(
                f"./logs/{configs['log_dir']}/best_model.eqx", model
            )

        print(f"train Loss: {running_loss / (i + 1)}")
        print(f"scale: {model.scale}")
        print(f"val Loss: {val_loss}")
        logger.info(
            f"train Loss: {running_loss / (i + 1)}, val Loss: {val_loss}"
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
