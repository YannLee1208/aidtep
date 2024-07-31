import logging
from typing import Optional, Literal
from loguru import logger

from aidtep.ml.criterien.L2 import L2Loss
from aidtep.ml.processor.processor import Processor
from aidtep.ml.models.base_models.torch_model import PyTorchModel
from aidtep.ml.data.dataloader import create_dataloaders
from aidtep.ml.utils.common import get_model_class, get_criterien, get_optimizer, get_scheduler
from aidtep.utils.initialize import initialize


class IAEAInverseBuilder:
    def __init__(self):
        pass

    def build_dataloaders(self, x_path: str, y_path: str, train_ratio: float, val_ratio: float, batch_size: int):
        logger.info("Building dataloaders")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(x_path, y_path, train_ratio,
                                                                                  val_ratio,
                                                                                  batch_size)
        logger.info("Dataloaders built")
        return self

    def build_model(self, model_type: str, criterien: str, optimizer: str, scheduler: str, lr: float,
                    device: Optional[Literal['cpu', 'cuda']]):
        logger.info(f"Buiding model of type {model_type}, criterien {criterien}, optimizer {optimizer}, scheduler {scheduler}, lr {lr}")
        model = get_model_class(model_type)()
        criterion = get_criterien(criterien)()
        optimizer = get_optimizer(optimizer)(model.parameters(), lr=lr)
        # TODO : design a better scheduler
        scheduler = get_scheduler(scheduler)(optimizer, step_size = 50, gamma = 0.1)
        self.model = PyTorchModel(model, criterion, optimizer, scheduler, device)

        logger.info("Adding criterien")
        self.model.add_criteria("L2", L2Loss())

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
    initialize(log_level=logging.DEBUG)
    # import torch
    # print(torch.cuda.is_available())
