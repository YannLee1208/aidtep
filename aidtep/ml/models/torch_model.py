import logging
from typing import Literal
import torch
from black import Optional
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from aidtep.ml.models.base_model import BaseModel


class PyTorchModel(BaseModel):
    """
    PyTorch model class for training and prediction.
    """

    def __init__(self, model: nn.Module, criterion, optimizer, scheduler=None,
                 device: Optional[Literal['cpu', 'cuda']] = None):
        """
        :param model: PyTorch model
        :param criterion: Loss function
        :param optimizer: Optimizer
        :param scheduler: Learning rate scheduler
        :param device (str): Device to run the model on (cpu or cuda)
        """
        super().__init__()
        self.scheduler = scheduler
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        logger.info(f"Using device: {self.device}")
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer

        self.addtional_criterias = []

    def add_criteria(self, criterion):
        self.addtional_criterias.append(criterion)

    def train(self, dataloader: DataLoader, **kwargs) -> float:
        """
        Train the model for one epoch.
        :param dataloader: DataLoader, containing training data
        :param kwargs:
        :return: epoch_loss
        """
        self.model.train()
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            epoch_loss += loss.item()
        return epoch_loss

    def predict(self, dataloader: DataLoader) -> torch.Tensor:
        self.logger.info("Predicting with PyTorch model...")
        self.model.eval()
        with torch.no_grad():
            for idx, (batch_x, batch_y) in enumerate(dataloader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                if idx == 0:
                    predictions = outputs
                else:
                    predictions = torch.cat((predictions, outputs))
        return predictions

    def evaluate(self, dataloader: DataLoader, **kwargs) -> float:
        self.logger.info("Evaluating PyTorch model...")
        self.model.eval()
        loss = 0.0
        with torch.no_grad():
            for idx, (batch_x, batch_y) in enumerate(dataloader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss += self.criterion(outputs, batch_y).item()
            return loss

    def save_model(self, filepath: str) -> None:
        self.logger.info(f"Saving PyTorch model to {filepath}...")
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath: str) -> None:
        self.logger.info(f"Loading PyTorch model from {filepath}...")
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
