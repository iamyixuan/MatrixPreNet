import re

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def set_size(width, fraction=1, subplots=(1, 1), golden_ratio=True):
    """Set figure dimensions to avoid scaling in LaTeX.
    from https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }

    plt.rcParams.update(tex_fonts)

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if golden_ratio:
        fig_height_in = (
            fig_width_in * golden_ratio * (subplots[0] / subplots[1])
        )
    else:
        fig_height_in = fig_width_in

    return (fig_width_in, fig_height_in)


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
    print(file_path)
    train_loss = []
    val_loss = []
    scale = []
    train_pattern = r"train Loss: (\d+\.\d+)"
    val_pattern = r"val Loss: (\d+\.\d+)"
    scale_pattern = r"scale: \[?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\]?"
    with open(file_path, "r") as f:
        for line in f:
            train_match = re.findall(train_pattern, line)
            val_match = re.findall(val_pattern, line)
            scale_match = re.findall(scale_pattern, line)
            if len(train_match) == 0 or len(val_match) == 0:
                continue
            train_loss.append(float(train_match[0]))
            val_loss.append(float(val_match[0]))
            scale.append(float(scale_match[0]))

    return train_loss, val_loss, scale


def plot_train(train_loss, val_loss):
    x = jnp.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=set_size("thesis"))
    ax.plot(x, train_loss, label="train loss")
    ax.plot(x, val_loss, label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(linestyle="dotted")
    ax.legend()
    return fig, ax


def plot_multi_curves(log_list, label_list):
    fig, ax = plt.subplots(figsize=set_size("thesis"))
    ax2 = ax.twinx()
    for i, log in enumerate(log_list):
        train_loss, val_loss, scale = read_log(log)
        x = jnp.arange(1, len(train_loss) + 1)
        ax.plot(x, train_loss, label=label_list[i])
        ax2.plot(x, scale, label=label_list[i] + " scale", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel(r"$\alpha$")
    # ax2.set_yscale("log")
    ax.grid(linestyle="dotted")
    ax.legend()
    return fig, ax


def plot_hist(hist_list, label_list):
    fig, ax = plt.subplots(figsize=set_size("thesis"))
    q90 = []
    for i, hist in enumerate(hist_list):
        q90.append(np.percentile(hist, 90))
        ax.hist(hist, bins=500, alpha=0.3, label=label_list[i])
    ax.set_xlabel("Condition number")
    ax.set_xlim(0, np.max(q90))
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig, ax

def plot_sorted_scatter(point_lists, labels):
    fig, ax = plt.subplots(figsize=set_size("thesis"))
    base_order = np.argsort(point_lists[0])
    q90 = []
    for i, points in enumerate(point_lists):
        q90.append(np.percentile(points, 90))
        ax.scatter(np.arange(len(points)), points[base_order], label=labels[i])
    ax.set_xlabel("Index")
    ax.set_ylabel("Condition number")
    ax.set_ylim(0, np.max(q90))
    ax.legend()
    return fig, ax


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    train_loss, val_loss, scale = read_log(args.log_path)
    fig, _ = plot_train(train_loss, val_loss)
    fig.savefig(
        args.log_path.replace(".txt", ".png"),
        dpi=500,
        bbox_inches="tight",
    )

    # log_list = [
    #     "./logs/U1_FNN_U_tilde/log.txt",
    #     "./logs/MPNodeEdgeModel_2_1_1_8_8_train_adam_ConditionNumberLoss_0.0001_100_128_True/train.log",
    #     "./logs/GraphMaskModel_2_1_2_3_3_1024_1792_train_adam_ConditionNumberLoss_0.0001_100_True/train.log",
    #     "./logs/model_2_1_4_3_3_3_2_train_adam_ConditionNumberLoss_0.0001_100_True/train.log",
    #     "./logs/U1_FNN_M_lr1e-4_subset200/log.txt",
    # ]
    # label_list = ["U1_tilde", "U1EdgeGNN", "U1GNN", "U1GNN_Gamma", "U1FNN"]
    # fig, ax = plot_multi_curves(log_list, label_list)
    # fig.savefig(
    #     "./logs/compare.png",
    #     dpi=500,
    #     bbox_inches="tight",
    # )
