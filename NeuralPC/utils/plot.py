import pickle

import matplotlib.pyplot as plt
import numpy as np


def plotTrainCurve(trainCurve, valCurve):
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["lines.linewidth"] = 2
    fig, ax = plt.subplots()
    ax.set_box_aspect(1 / 1.62)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.plot(trainCurve, label="Train")
    ax.plot(valCurve, label="Validation")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    return fig


if __name__ == "__main__":
    with open(
        "../../logs/2024-03-19-19_DDApprox_MAEtrainLog.pkl",
        "rb",
    ) as f:
        log = pickle.load(f)

    trainCurve = log["TrainLoss"]
    valCurve = log["ValLoss"]
    fig = plotTrainCurve(trainCurve, valCurve)
    fig.savefig(
        "../../figures/explicitConNum_DDStructure_trainLog.pdf", format="pdf", bbox_inches="tight"
    )
    plt.show()
