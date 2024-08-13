from typing import Optional
from loguru import logger
import torch

from aidtep.ml.criterion.L2 import L2Loss
from aidtep.ml.processor.processor import Processor
from aidtep.ml.models.base_models.torch_model import PyTorchModel
from aidtep.ml.data.dataloader import create_dataloaders
from aidtep.ml.scheduler import get_scheduler_class
from aidtep.ml.optimizer import get_optimizer_class
from aidtep.ml.criterion import get_criterion_class
from aidtep.ml.models import get_model_class


class IAEAForwardBuilder:
    def __init__(self):
        pass

    def build_dataloaders(self, x_path: str, y_path: str, train_ratio: float, val_ratio: float, batch_size: int):
        logger.info("Building dataloaders")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(x_path, y_path, train_ratio,
                                                                                  val_ratio,
                                                                                  batch_size)
        logger.info("Dataloaders built")
        return self

    def build_model(self, model_type: str, criterion_type: str, optimizer_type: str, scheduler_type: str, lr: float,
                    device: torch.device, criterion_args: Optional[dict] = None, optimizer_args: Optional[dict] = None, scheduler_args: Optional[dict] = None):
        logger.info(f"Building model of type {model_type}, criterion {criterion_type}, optimizer {optimizer_type}, scheduler {scheduler_type}, lr {lr}")
        model = get_model_class(model_type)()
        criterion = get_criterion_class(criterion_type)(**criterion_args)
        optimizer = get_optimizer_class(optimizer_type)(model.parameters(), lr=lr, **optimizer_args)
        scheduler = get_scheduler_class(scheduler_type)(optimizer, **scheduler_args)
        self.model = PyTorchModel(model, criterion, optimizer, scheduler, device)

        logger.info("Adding forward-specific criteria (if any)")
        # Add any additional criteria specific to forward problem if needed.

        logger.info("Model built")
        return self

    def train(self, epochs: int, model_path: str):
        logger.info("Starting training")
        processor = Processor(self.model)
        processor.train(self.train_loader, self.val_loader, epochs, model_path)
        logger.info("Training done")

    def test(self, model_path: str):
        self.model.load_model(model_path)
        processor = Processor(self.model)
        return processor.test(self.test_loader)

if __name__ == '__main__':
    forward_problem = IAEAForwardBuilder()
    forward_problem.build_dataloaders(x_path="data/x_train.csv", y_path="data/y_train.csv", train_ratio=0.7, val_ratio=0.2, batch_size=32)
    forward_problem.build_model(model_type="NVT_ResNet", criterion_type="mse", optimizer_type="adam", scheduler_type="StepLR", lr=0.001, device=torch.device("cuda"))
    forward_problem.train(epochs=10, model_path="models/forward_model.pth")
    test_results = forward_problem.test(model_path="models/forward_model.pth")
    print(test_results)
