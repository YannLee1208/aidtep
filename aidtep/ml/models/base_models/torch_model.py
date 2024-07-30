from typing import Literal, Optional
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from aidtep.ml.models.base_models.base_model import BaseModel
from aidtep.ml.utils.log import ProgressLogger
from aidtep.utils.file import make_parent_dir, check_file_exist


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

        self.additional_criterions = {}
        self.progress_logger = ProgressLogger()

    def add_criteria(self, name, criterion):
        self.additional_criterions[name] = criterion

    def train(self, dataloader: DataLoader, progress_prefix: str = "", **kwargs) -> float:
        """
        Train the model for one epoch.
        :param progress_prefix: str, prefix for the progress logger
        :param dataloader: DataLoader, containing training data
        :return: total loss of dataloader
        """
        self.model.train()
        epoch_loss = 0.0
        batch_number = len(dataloader)

        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            with torch.no_grad():
                criterion_losses = {"train_loss": loss.item()}
                for name, criterion in self.additional_criterions.items():
                    criterion_losses[name] = criterion(outputs, targets).item()
                self.progress_logger.log_batch(progress_prefix, idx + 1, batch_number, criterion_losses)
        self.progress_logger.finalize_log()

        if self.scheduler is not None:
            self.scheduler.step()
        return epoch_loss

    def predict(self, dataloader: DataLoader) -> torch.Tensor:
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

    def evaluate(self, dataloader: DataLoader, progress_prefix : str = "", **kwargs) -> float:
        """
        Evaluate the model on the given dataloader.
        :param dataloader: DataLoader, containing evaluation data
        :param progress_prefix: str, prefix for the progress logger
        :return: total loss of dataloader
        """
        self.model.eval()
        total_loss = 0.0
        batch_number = len(dataloader)
        with torch.no_grad():
            for idx, (batch_x, batch_y) in enumerate(dataloader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                criterion_losses = {"train_loss": loss.item()}
                for name, criterion in self.additional_criterions.items():
                    criterion_losses[name] = criterion(outputs, batch_y).item()
                self.progress_logger.log_batch(progress_prefix, idx + 1, batch_number, criterion_losses)
            self.progress_logger.finalize_log()
            return total_loss

    def save_model(self, filepath: str) -> None:
        """
        Save the PyTorch model to the given filepath.
        :param filepath: str, path to save the model
        :return: None
        """
        make_parent_dir(filepath)
        self.logger.info(f"Saving PyTorch model to {filepath}...")
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load the PyTorch model from the given filepath.
        :param filepath: str, path to load the model
        :return: None
        """
        if not check_file_exist(filepath):
            raise ValueError(f"File not found: {filepath}")
        self.logger.info(f"Loading PyTorch model from {filepath}...")
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()

    def log_memory_usage(self):
        """
        Log the current memory usage of the GPU.
        """
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        logger.info(f"Memory allocated: {allocated:.2f} MB")
