import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..utils.pair_data import U1Data, get_dataset
from ._base_trainer import BaseTrainer


class UnsupervisedTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        **kwargs,
    ):
        super(UnsupervisedTrainer, self).__init__(
            model,
            optimizer,
            criterion,
            **kwargs,
        )

        self.kwargs = kwargs
        self.criterion = criterion

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
            dataset = U1Data(
                data_dir,
                batch_size,
                shuffle=True,
                validation_split=0.2,
            )
        train_loader, self.val_loader = dataset.get_data_loader()

        optimizer = self.optimizer(
            self.model.parameters(), lr=self.kwargs["learning_rate"]
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_loss = 1e10
        pbar = tqdm(range(num_epochs), desc="Training")
        for epoch in pbar:
            self.model.train()
            running_loss = 0.0
            for i, train_data in enumerate(train_loader, 0):
                train_inputs, _ = train_data
                train_inputs = train_inputs.to(self.device)
                optimizer.zero_grad()
                train_outputs = self.model(train_inputs)
                # print(torch.norm(outputs, dim=0).mean(), "outputs norm")
                # print(torch.norm(inputs, dim=0).mean(), "inputs norm")
                # assert False
                loss = self.criterion(
                    train_inputs, train_outputs, scale=self.model.scale
                )  # learn the inverse map
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # grads = torch.stack(
                #         [
                #             torch.norm(param.grad)
                #             for param in self.model.parameters()
                #         ]
                #     )
                # print("grads shape", grads.shape)
                # gradient_norm_max = torch.norm(grads)
                # print(f"Gradient min: {gradient_norm_max}")
                # if i > 10:
                #     assert False
            train_loss = running_loss / i
            scheduler.step()
            self.log(f"Epoch {epoch+1}, loss: {train_loss}")
            running_loss = 0.0

            scale_after_training = self.model.scale.item()

            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for i, val_data in enumerate(self.val_loader, 0):
                    val_inputs, _ = val_data
                    val_inputs = val_inputs.to(self.device)
                    val_outputs = self.model(val_inputs)

                    val_ls = self.criterion(
                        val_inputs, val_outputs, scale=self.model.scale
                    )

                    assert scale_after_training == self.model.scale.item()
                    val_loss += val_ls.item()
                self.log(f"Validation loss: {val_loss}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model()
            pbar.set_postfix(
                {
                    "train_loss": f"{train_loss:.3f}",
                    "val_loss": f"{val_loss:.3f}",
                    "scale": f"{self.model.scale.item():.3f}",
                }
            )

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
            "inputs": [],
            "pred": [],
            "train_inputs": [],
            "train_pred": [],
            "true": [],
            "train_true": [],
        }
        with torch.no_grad():
            for data in val_loader:
                self.model.eval()
                x, _ = data
                x = x.to(self.device)
                out = self.model(x)
                pred_results["pred"].append(out.cpu())
                pred_results["inputs"].append(x.cpu())
                pred_results["true"].append(_.cpu())
            for data in train_loader:
                x, _ = data
                x = x.to(self.device)
                out = self.model(x)
                pred_results["train_pred"].append(out.cpu())
                pred_results["train_inputs"].append(x.cpu())
                pred_results["train_true"].append(_.cpu())
        return pred_results
