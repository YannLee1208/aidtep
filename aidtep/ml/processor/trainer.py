import numpy as np
from torch.utils.data import DataLoader
from loguru import logger

from aidtep.ml.models.torch_model import PyTorchModel


class Trainer:
    def __init__(self, model: PyTorchModel):
        self.model = model

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        best_val_loss = np.inf
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            train_loss = self.model.train(train_loader)
            val_loss = self.model.evaluate(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            logger.info(f"EPOCH: {epoch} | TRAIN LOSS: {train_loss} | VAL LOSS: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_model(f"model_best.pth")

        return train_losses, val_losses
