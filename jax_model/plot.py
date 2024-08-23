import re

import jax.numpy as jnp
import matplotlib.pyplot as plt


def read_log(file_path):
    train_loss = []
    val_loss = []
    train_pattern = r"train Loss: (\d+\.\d+)"
    val_pattern = r"val Loss: (\d+\.\d+)"
    with open(file_path, "r") as f:
        for line in f:
            train_match = re.findall(train_pattern, line)
            val_match = re.findall(val_pattern, line)
            if len(train_match) == 0 or len(val_match) == 0:
                continue
            train_loss.append(float(train_match[0]))
            val_loss.append(float(val_match[0]))

    return train_loss, val_loss


def plot_train(train_loss, val_loss):
    x = jnp.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots()
    ax.set_box_aspect(1 / 1.62)
    ax.plot(x, train_loss, label="train loss")
    ax.plot(x, val_loss, label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig


path = "./logs/10_5_16_32_conditionNumber/train_logs.txt"
train_loss, val_loss = read_log(path)
fig = plot_train(train_loss, val_loss)
fig.savefig(
    "/Users/yixuan.sun/Documents/projects/Preconditioners/docs/updates/figs/precondCNN_cond_traincurve.png",
    dpi=500,
    bbox_inches="tight",
)
