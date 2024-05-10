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
        train_pattern = re.compile(r"Epoch \d+, loss: \d+")
        val_pattern = re.compile(r"Validation loss: \d+")
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
        cbar = ax.figure.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
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
        ax.set_box_aspect(1 / 3)
        ax.scatter(
            org_e,
            0.1 * torch.ones_like(org_e),
            marker="x",
            color="r",
            label="original-Lanczos",
        )
        ax.yaxis.set_visible(False)
        ax.scatter(
            pc_e,
            0.3 * torch.ones_like(pc_e),
            marker="x",
            color="b",
            label="preconditioned-Lanczos",
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
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2)
        plt.show()
        return fig


if __name__ == "__main__":
    import pickle

    with open(
        "./logs/2024-04-25-19_LL_T_left_pc_spectrum_loss_penalized.pkl", "rb"
    ) as f:
        logger = pickle.load(f)

    with open("./logs/default_cg_solve_20runs.pkl", "rb") as f:
        info_list = pickle.load(f)

    with open("./logs/npc_cg_solve.pkl", "rb") as f:
        npc_info = pickle.load(f)

    with open("./logs/spectrum_npc_cg_solve.pkl", "rb") as f:
        spectrum_npc_info = pickle.load(f)

    with open(
        "./logs/best_LL_T_left_pc_true_condNum_loss_penalized_5_model-test-spectrum.pkl",
        "rb",
    ) as f:
        spectra = pickle.load(f)

    # generate random idx within 100

    idx = np.random.randint(0, 100)
    org_e = spectra["org"].detach()[idx]
    pc_e = spectra["pc"].detach()[idx]
    org_true_e = spectra["org_true"].detach()[idx]
    pc_true_e = spectra["pc_true"].detach()[idx]
    lower_L_e = spectra["lower_L"].detach()[idx]
    mat = spectra["L"].detach().abs()[idx]

    residual_hist = [info["residuals"] for info in info_list]
    npc_residual_hist = [res.detach().numpy() for res in npc_info["residuals"]]
    spectrum_npc_residual_hist = [
        res.detach().numpy() for res in spectrum_npc_info["residuals"]
    ]
    plotter = Plotter()

    linear_inv_logger_path = "./logs/train_logs/linear_inverse-linear_inverse_singleU1-1000-B32-lr0.0001.log"

    logger = plotter.read_logger(linear_inv_logger_path)
    train_curve = plotter.train_curve(logger, if_log=False)
    train_curve.savefig(
        "./figures/linear_inverse_single-U1-train_curve.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    # spec_fig = plotter.plot_spectrum(
    #     org_e, pc_e, org_true_e=org_true_e, pc_true_e=pc_true_e,  lower_L=lower_L_e
    # )
    # mat_fig = plotter.vis_matrix(mat)
    # spec_fig.savefig(
    #     "./figures/spectrum-condNum_loss_penalized_true_spectrum_log.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    # )
    # mat_fig.savefig(
    #     "./figures/spectrum_loss-condNum-L_mat.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    # )

    # fig = plotter.plot_cg_convergence(
    #     residual_hist, npc_residual_hist, spectrum_npc_residual_hist,
    #     npc_names=["npc", "spectrum_npc"],
    #     log_scale=True
    # )
    # train_curve = plotter.train_curve(logger, if_log=True)

    # train_curve.savefig(
    #     "./figures/LL_spectrum_penalized_train_curve.png",
    #     format="png",
    #     dpi=150,
    #     bbox_inches="tight",
    # )
    # fig.savefig(
    #     "./figures/npc_cg_spectrum_convergence.png",
    #     format="png",
    #     dpi=150,
    #     bbox_inches="tight",
    # )
