import matplotlib.pyplot as plt
import numpy as np
import torch


class Plotter:
    # plt.rcParams["lines.linewidth"] = 3
    # plt.rcParams["font.size"] = 14
    # plt.rcParams["axes.linewidth"] = 1

    def __init__(self):
        fontsize = 14
        plt.rcParams.update(
            {
                "figure.figsize": self.figure_size(246.0 * 2, fraction=1),
                "figure.facecolor": "white",
                "figure.edgecolor": "white",
                "savefig.dpi": 360,
                "figure.subplot.bottom": 0.5,
                # Use LaTeX to write all text
                # "text.usetex": True,
                "font.family": "serif",
                # Use 10pt font in plots, to match 10pt font in document
                "axes.labelsize": fontsize,
                "font.size": fontsize,
                # Make the legend/label fonts a little smaller
                "legend.fontsize": fontsize - 3,
                "xtick.labelsize": fontsize - 1,
                "ytick.labelsize": fontsize - 1,
                # tight layout,
                "figure.autolayout": True,
            }
        )

    def figure_size(self, width, fraction=1):
        """Set figure dimensions to avoid scaling in LaTeX.

        Args:
            width (float): Document textwidth or columnwidth in pts.
            fraction (float, optional) Fraction of the width which you wish the figure to occupy.

        Returns:
            tuple: Dimensions of figure in inches.
        """
        # Width of figure (in pts)
        fig_width_pt = width * fraction
        # Convert from pt to inches
        inches_per_pt = 1 / 72.27

        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        golden_ratio = (5**0.5 - 1) / 2

        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio

        fig_dim = (fig_width_in, fig_height_in)

        return fig_dim

    def read_logger(self, path):
        import re

        train_loss = []
        val_loss = []
        train_pattern = re.compile(r"Epoch \d+, loss: \d+\.\d+")
        val_pattern = re.compile(r"Validation loss: \d+\.\d+")
        with open(path, "r") as f:
            for line in f:
                matches_train = train_pattern.findall(line)
                matches_val = val_pattern.findall(line)
                if matches_train:
                    train_loss.append(float(matches_train[0].split()[-1]))
                elif matches_val:
                    val_loss.append(float(matches_val[0].split()[-1]))
        logger = {
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        return logger

    def train_curve(self, logger, if_log=False):
        if "train_obj_loss" in logger.keys():

            train_obj = logger["train_obj_loss"]
            val_obj = logger["val_obj_loss"]
            train_basis = logger["train_basis_loss"]
            val_basis = logger["val_basis_loss"]
            fig, axs = plt.subplots(1, 2)
            for ax in axs:
                ax.grid(True, linestyle="dotted", linewidth=0.5)
                ax.set_box_aspect(1 / 1.62)
                ax.set_xlabel("Epochs")
                if if_log:
                    ax.set_yscale("log")
            axs[0].plot(train_obj, label="train obj loss")
            axs[0].plot(val_obj, label="val obj loss")
            axs[0].set_ylabel("objective loss")
            axs[1].plot(train_basis, label="train ortho loss")
            axs[1].plot(val_basis, label="val ortho loss")
            axs[1].set_ylabel("orthogonality loss")
            axs[0].legend()
            axs[1].legend()

            plt.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots()
            ax.plot(logger["train_loss"], label="train loss")
            ax.plot(logger["val_loss"], label="val loss")
            ax.set_xlabel("Epochs")
            ax.legend()
            if if_log:
                ax.set_yscale("log")
        ax.set_xlim(0, 3000)
        return fig

    def scatter_plot(self, true, pred):
        fig, ax = plt.subplots()
        ax.scatter(true, pred)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        plt.show()
        return fig

    def vis_matrix(self, matrix, title=None):
        fig, ax = plt.subplots()
        ax.imshow(matrix, cmap="viridis")
        if title:
            ax.set_title(title)

        # add a colorbar and make the height the same as the figure
        cbar = ax.figure.colorbar(
            ax.images[0], ax=ax, fraction=0.046, pad=0.04
        )
        cbar.ax.set_ylabel("entry absolute values")
        plt.show()

        return fig

    def plot_cg_convergence(self, residual_hist, *args, **kwarg):
        fig, ax = plt.subplots()
        ax.set_box_aspect(1 / 1.62)
        residual_hist = np.array(residual_hist)
        if residual_hist.ndim == 1:
            ax.plot(residual_hist)
        else:
            iters = np.arange(0, residual_hist.shape[1])
            residual_mean = residual_hist.mean(axis=0)
            residual_std = residual_hist.std(axis=0)
            ax.plot(iters, residual_mean, color="b", label="defualt CG")
            ax.scatter(20, 1.93, color="r", label="with NN PC (val.)")
            ax.scatter(20, 1.38, color="green", label="with NN PC (train)")
            ax.vlines(
                20,
                ymin=0,
                ymax=residual_mean[20],
                color="gray",
                linestyle="dotted",
                linewidth=1.5,
            )
            ax.hlines(
                residual_mean[20],
                xmin=0,
                xmax=20,
                color="gray",
                linestyle="dotted",
                linewidth=1.5,
            )
            ax.set_xticks(np.arange(0, residual_mean.shape[0], 10))
            ax.fill_between(
                iters,
                residual_mean - residual_std,
                residual_mean + residual_std,
                alpha=0.2,
                color="b",
            )
            ax.hlines(
                1.93,
                xmin=0,
                xmax=20,
                color="r",
                linestyle="dotted",
                linewidth=1.5,
            )
            ax.hlines(
                1.38,
                xmin=0,
                xmax=20,
                color="green",
                linestyle="dotted",
                linewidth=1.5,
            )

        if args:
            for i, arg in enumerate(args):
                ax.plot(arg, label=f"npc {kwarg['npc_names'][i]}")

        if kwarg["log_scale"]:
            ax.set_yscale("log")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual Norm")
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=0)
        ax.legend()
        plt.show()
        return fig

    def plot_spectrum(self, org_e, pc_e, **kwarg):
        fig, ax = plt.subplots()
        cond_DD = org_e.max() / org_e.min()
        cond_pc = pc_e.max() / pc_e.min()
        ax.set_box_aspect(1 / 3)
        ax.scatter(
            org_e,
            0.1 * torch.ones_like(org_e),
            marker="x",
            color="r",
            label=r"$D^{\dagger}D$",
        )
        ax.yaxis.set_visible(False)
        ax.scatter(
            pc_e,
            0.2 * torch.ones_like(pc_e),
            marker="x",
            color="b",
            label=r"$LD^{\dagger}DL^{\dagger}$",
        )

        if kwarg.get("train_org_e") is not None:
            ax.scatter(
                kwarg["train_org_e"],
                0.3 * torch.ones_like(kwarg["train_org_e"]),
                marker=".",
                color="r",
                label="train-original",
            )
            ax.scatter(
                kwarg["train_pc_e"],
                0.4 * torch.ones_like(kwarg["train_pc_e"]),
                marker=".",
                color="b",
                label="train-preconditioned",
            )
            cond_DD_train = max(
                cond_DD,
                kwarg["train_org_e"].max() / kwarg["train_org_e"].min(),
            )
            cond_pc_train = max(
                cond_pc, kwarg["train_pc_e"].max() / kwarg["train_pc_e"].min()
            )

        if kwarg.get("org_true_e") is not None:
            ax.scatter(
                kwarg["org_true_e"],
                0.2 * torch.ones_like(kwarg["org_true_e"]),
                marker=".",
                color="r",
                label="original-true",
            )
        if kwarg.get("pc_true_e") is not None:
            ax.scatter(
                kwarg["pc_true_e"],
                0.4 * torch.ones_like(kwarg["pc_true_e"]),
                marker=".",
                color="b",
                label="preconditioned-true",
            )

        if kwarg.get("lower_L") is not None:
            ax.scatter(
                kwarg["lower_L"],
                0.5 * torch.ones_like(kwarg["lower_L"]),
                marker=".",
                color="g",
                label="lower_L",
            )

        # log scale for x
        ax.set_xscale("log")
        ax.set_ylim(0.0, 0.6)
        ax.text(
            1e-3,
            0.54,
            f"[val] Condition numbers: {cond_DD:.2f}, {cond_pc:.2f}",
        )
        ax.text(
            1e-3,
            0.51,
            f"[train] Condition numbers: {cond_DD_train:.2f}, {cond_pc_train:.2f}",
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2)
        plt.show()
        return fig


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_type", type=str, help="plot type")
    parser.add_argument("--file_path", type=str, help="file path")
    args = parser.parse_args()

    plotter = Plotter()
    if args.plot_type == "train_curve":
        logger = plotter.read_logger(args.file_path + "model-train.log")
        train_curve = plotter.train_curve(logger, if_log=True)
        train_curve.savefig(
            f"{args.file_path}/train_curves_log.pdf",
            format="pdf",
            bbox_inches="tight",
        )
    elif args.plot_type == "spectrum":
        with open(args.file_path + "val_pred.pkl", "rb") as f:
            data = pickle.load(f)

        DD = data["inputs"][0][0].squeeze()
        pred = data["pred"][0][0].squeeze()

        train_inputs = data["train_inputs"][0][0].squeeze()
        train_pred = data["train_pred"][0][0].squeeze()
        train_precond = torch.matmul(
            torch.matmul(DD, train_pred), train_pred.conj().T
        )
        train_org_e = torch.linalg.eigvals(DD).real
        train_pc_e = torch.linalg.eigvals(train_precond).real

        precond = torch.matmul(torch.matmul(DD, pred), pred.conj().T)
        org_e = torch.linalg.eigvals(DD).real
        pc_e = torch.linalg.eigvals(precond).real

        fig = plotter.plot_spectrum(
            org_e, pc_e, train_org_e=train_org_e, train_pc_e=train_pc_e
        )
        fig.savefig(
            f"{args.file_path}/spectrum.pdf",
            format="pdf",
            bbox_inches="tight",
        )
