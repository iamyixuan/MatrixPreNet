import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    plt.rcParams["lines.linewidth"] = 3
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.linewidth"] = 1

    def train_curve(self, logger):

        train_obj = logger["train_obj_loss"]
        val_obj = logger["val_obj_loss"]
        train_basis = logger["train_basis_loss"]
        val_basis = logger["val_basis_loss"]
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        for ax in axs:
            ax.grid(True, linestyle="dotted", linewidth=0.5)
            ax.set_box_aspect(1 / 1.62)
            ax.set_xlabel("Epochs")
        axs[0].plot(train_obj, label="train obj loss")
        axs[0].plot(val_obj, label="val obj loss")
        axs[0].set_ylabel("L2 norm of residuals")
        axs[1].plot(train_basis, label="train ortho loss")
        axs[1].plot(val_basis, label="val ortho loss")
        axs[1].set_ylabel("orthogonality loss")
        axs[0].legend()
        axs[1].legend()

        plt.tight_layout()
        plt.show()
        return fig

    def scatter_plot(self, true, pred):
        fig, ax = plt.subplots()
        ax.scatter(true, pred)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        plt.show()
        return fig

    def plot_cg_convergence(self, residual_hist, *args):
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
            # ax.scatter(20, 53.1180, color="r", label="with NN PC (val.)")
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
            # ax.hlines(
            #     53.1180,
            #     xmin=0,
            #     xmax=20,
            #     color="r",
            #     linestyle="dotted",
            #     linewidth=1.5,
            # )

        if args:
            for arg in args:
                ax.plot(arg, color="b")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual Norm")
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=0, xmax=60)
        ax.legend()
        plt.show()
        return fig


if __name__ == "__main__":
    import pickle

    with open("./logs/2024-04-17-14_K-cond_TwoSide_basisOrtho.pkl", "rb") as f:
        logger = pickle.load(f)

    with open("./logs/default_cg_solve_20runs.pkl", "rb") as f:
        info_list = pickle.load(f)

    residual_hist = [info["residuals"] for info in info_list]
    plotter = Plotter()
    fig = plotter.plot_cg_convergence(residual_hist)
    # train_curve = plotter.train_curve(logger)
    # train_curve.savefig(
    #     "./figures/LL_train_curve.png",
    #     format="png",
    #     dpi=150,
    #     bbox_inches="tight",
    # )
    fig.savefig(
        "./figures/default_cg_convergence.png",
        format="png",
        dpi=150,
        bbox_inches="tight",
    )
