import logging
import os

import equinox as eqx
import jax
import jax.numpy as jnp
from box import Box
from src.model.GraphNets import MPNodeEdgeModel
from src.utils.data import U1DDatasetCOO
from src.utils.train import train_COO
from torch.utils.data import DataLoader

logging.getLogger(__name__)


def main(configs):
    if configs.double_precision:
        jax.config.update("jax_enable_x64", True)

    model_config = [str(m) for m in list(configs.model.values())]
    train_config = [str(t) for t in list(configs.train.values())]
    print(f"Model config: {model_config}")
    print(f"Train config: {train_config}")
    logname = "_".join(["MPNodeEdgeModel"] + model_config + ["train"] + train_config)

    os.makedirs(f"./logs/{logname}", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=f"./logs/{logname}/train.log",
        filemode="w",
    )
    train_data = U1DDatasetCOO(configs.data.path, mode="train")
    valid_data = U1DDatasetCOO(configs.data.path, mode="val")
    # test_data = U1DDataset(configs.data_path, mode="test")

    model = MPNodeEdgeModel(**configs.model, key=jax.random.PRNGKey(0))

    configs.train.logname = logname
    trained_model = train_COO(
        model=model,
        trainset=train_data,
        valset=valid_data,
        configs=configs.train,
    )

    if configs.train.save_model:
        eqx.tree_serialise_leaves(f"./logs/{logname}/last.eqx", trained_model)


if __name__ == "__main__":
    configs = Box.from_yaml(filename="./configs/MPGNN_config.yaml")
    main(configs)
