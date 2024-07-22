from typing import Literal

import torch
from torch import nn

from aidtep.ml.models.base_model import BaseModel


class PyTorchModel(BaseModel):
    """
    PyTorch model class for training and prediction.
    """
    def __init__(self, model: nn.Module, criterion, optimizer, device: Literal['cpu', 'cuda'] = 'cpu'):
        """
        :param model: PyTorch model
        :param criterion: Loss function
        :param optimizer: Optimizer
        :param device (str): Device to run the model on (cpu or cuda)
        """
        super().__init__()
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 10, batch_size: int = 32, **kwargs) -> None:
        self.logger.info("Training PyTorch model...")
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}")
        self.logger.info("Training completed.")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.logger.info("Predicting with PyTorch model...")
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            return self.model(X)

    def evaluate(self, X: torch.Tensor, y: torch.Tensor, **kwargs) -> float:
        self.logger.info("Evaluating PyTorch model...")
        self.model.eval()
        X, y = X.to(self.device), y.to(self.device)
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            return loss.item()

    def save_model(self, filepath: str) -> None:
        self.logger.info(f"Saving PyTorch model to {filepath}...")
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath: str) -> None:
        self.logger.info(f"Loading PyTorch model from {filepath}...")
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        
