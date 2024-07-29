import numpy as np
import torch
from torch.utils.data import DataLoader

from aidtep.ml.models.torch_model import PyTorchModel


class Trainer:
    def __init__(self, model: PyTorchModel):
        self.model = model

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int,
              model_path: str = None):
        """
        Train the model, save the best model based on validation loss
        :param train_loader: DataLoader for training data
        :param val_loader: DataLoader for validation data
        :param epochs: int, number of epochs to train
        :param model_path: str, path to save the best model
        :return: None
        """
        best_val_loss = np.inf
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            train_loss = self.model.train(train_loader, progress_prefix=f"Epoch {epoch+1}: ")
            val_loss = self.model.evaluate(val_loader, progress_prefix=f"Validation: ")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if model_path:
                    self.model.save_model(model_path)
            torch.cuda.empty_cache()

        return train_losses, val_losses
