import re
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def set_figuresize(fig, width, height, golden_ratio=False):
    if golden_ratio:
        width = height * (1 + np.sqrt(5)) / 2
    fig.set_figwidth(width)
    fig.set_figheight(height)

    # set font size based on figure size
    if width >= 10:
        font_size = 16
    elif width >= 8:
        font_size = 14
    else:
        font_size = 12
    plt.rcParams.update({"font.size": font_size})
    plt.rcParams.update({"axes.labelsize": font_size})

    # set line width based on figure size
    if width >= 10:
        line_width = 3
    elif width >= 8:
        line_width = 2.5
    else:
        line_width = 1.5
    plt.rcParams.update({"lines.linewidth": line_width})
    return fig

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
    fig = set_figuresize(fig, 1, 4, golden_ratio=True)
    ax.plot(x, train_loss, label="train loss")
    ax.plot(x, val_loss, label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    train_loss, val_loss = read_log(args.log_path)
    fig = plot_train(train_loss, val_loss)
    fig.savefig(
        args.log_path.replace(".log", ".png"),
        dpi=500,
        bbox_inches="tight",
    )
