import numpy as np
import pickle
import os
from datetime import datetime


class Logger:
    def __init__(self) -> None:
        self.logger = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }

    def record(self, key, value):
        if key not in self.logger.keys():
            self.logger[key] = []
        self.logger[key].append(value)

    def print(self):
        print(
            "Epoch: %d , Training loss: %.4f, Validation loss: %.4f"
            % (
                self.logger["epoch"][-1],
                self.logger["train_loss"][-1],
                self.logger["val_loss"][-1],
            )
        )

    def save(self, path):
        if not os.path.exists(path + "/logs/"):
            os.makedirs(path + "/logs/")
        with open(path + "./logs/logs.pkl", "wb") as f:
            pickle.dump(self.logger, f)
        print("Saving the training logs...")
