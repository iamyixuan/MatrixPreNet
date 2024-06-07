import torch

from ..utils.pair_data import PairDataset, get_dataset
from ._base_trainer import BaseTrainer
from tqdm import tqdm


class SupervisedTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        **kwargs,
    ):
        super(SupervisedTrainer, self).__init__(
            model,
            optimizer,
            criterion,
            **kwargs,
        )

        self.kwargs = kwargs
        self.criterion = criterion()


    def train(self, num_epochs, batch_size, learning_rate):
        data_dir = self.kwargs["data_dir"]
        if self.kwargs["data_name"] is not None:
            dataset = get_dataset(self.kwargs["data_name"])(
                data_dir,
                batch_size,
                shuffle=True,
                validation_split=0.2,
            )
        else:
            dataset = PairDataset(
                data_dir,
                batch_size,
                shuffle=True,
                validation_split=0.2,
            )
        train_loader, val_loader = dataset.get_data_loader()

        optimizer = self.optimizer(
            self.model.parameters(), lr=self.kwargs["learning_rate"]
        )

        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            self.log(f"Epoch {epoch+1}, loss: {running_loss/i}")
            running_loss = 0.0

            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for i, data in enumerate(val_loader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(
                        self.device
                    )
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                self.log(f"Validation loss: {val_loss}")

    def predict(self, checkpoint=None):
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint))

        dataset = get_dataset(self.kwargs["data_name"])(
            self.kwargs["data_dir"],
            self.kwargs["batch_size"],
            shuffle=True,
            validation_split=0.2,
        )
        train_loader, val_loader = dataset.get_data_loader()
        pred_results = {
            "true": [],
            "pred": [],
            "train_true": [],
            "train_pred": [],
            "inputs": [],
            "train_inputs": [],
        }
        with torch.no_grad():
            for x, y in val_loader:
                self.model.eval()
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x)
                pred_results["pred"].append(out.cpu())
                pred_results["true"].append(y.cpu())
                pred_results["inputs"].append(x.cpu())
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x)
                pred_results["train_pred"].append(out.cpu())
                pred_results["train_true"].append(y.cpu())
                pred_results["train_inputs"].append(x.cpu())
        return pred_results
