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
        self.model.to(self.device)
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
        additional_info = kwargs.get("additional_info", "")

        log_name = f"{model}-{data}-{epochs}-B{batch_size}-lr{learning_rate}-{additional_info}"
        self.log_path = f"./experiments/{log_name}/"

        # create directory if it does not exist
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename=self.log_path + "model-train.log",  # Log file path
            filemode="a",
        )
        print(f"Trainer created. Logging to {self.log_path}")
        return

    @abstractmethod
    def train(self):
        pass

    def log(self, message):
        self.logger.info(message)

    def save_model(self):
        torch.save(self.model.state_dict(), self.log_path + "model.pth")
