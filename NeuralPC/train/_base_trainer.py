import logging
import os
from abc import ABC, abstractmethod

import torch


class BaseTrainer(ABC):
    def __init__(self, model, optimizer, criterion, **info_kwargs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.info_kwargs = info_kwargs
        self.init(**info_kwargs)

    def init(self, **kwargs):
        assert kwargs.get("model_type") is not None, "Model type not provided"
        model = kwargs["model_type"]
        data = kwargs["data_name"]
        epochs = kwargs["epochs"]
        batch_size = kwargs["batch_size"]
        learning_rate = kwargs["learning_rate"]

        log_name = f"{model}-{data}-{epochs}-B{batch_size}-lr{learning_rate}"
        log_path = f"./logs/train_logs/{log_name}/"

        # create directory if it does not exist
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename=log_path + "model-train.log",  # Log file path
            filemode="w",  # Overwrite the log file
        )
        print(f"Trainer created. Logging to {log_path}")
        return

    @abstractmethod
    def train(self):
        pass

    def log(self, message):
        self.logger.info(message)

    def save_model(self, path):
        model = self.info_kwargs["model"]
        data = self.info_kwargs["data"]
        epochs = self.info_kwargs["epochs"]
        batch_size = self.info_kwargs["batch_size"]
        learning_rate = self.info_kwargs["learning_rate"]

        save_model_name = (
            f"{model}-{data}-{epochs}-B{batch_size}-lr{learning_rate}.pth"
        )
        model_path = f"./logs/saved_models/{save_model_name}"

        torch.save(self.model.state_dict(), model_path)
