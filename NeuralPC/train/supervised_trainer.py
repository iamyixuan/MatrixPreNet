from ..utils.pair_data import PairDataset
from ._base_trainer import BaseTrainer
import torch


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
        dataset = PairDataset(
            data_dir,
            batch_size,
            shuffle=True,
            validation_split=0.2,
        )
        train_loader, val_loader = dataset.get_data_loader()

        optimizer = self.optimizer(self.model.parameters(), lr=self.kwargs["learning_rate"])

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(labels)
                loss = self.criterion(outputs, inputs) # learn the inverse map
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

