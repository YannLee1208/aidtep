import os
from abc import ABC
from typing import Optional

import torch
from loguru import logger

from aidtep.ml.criterion import get_criterion_class
from aidtep.ml.criterion.L2 import L2Loss, LinfLoss
from aidtep.ml.data.dataloader import create_dataloaders
from aidtep.ml.models import get_model_class
from aidtep.ml.models.base_models.torch_model import PyTorchModel
from aidtep.ml.optimizer import get_optimizer_class
from aidtep.ml.processor.processor import Processor
from aidtep.ml.scheduler import get_scheduler_class
from aidtep.utils.common import Registry, import_modules


class InverseBuilder(Registry, ABC):
    builder_mapping = {}

    def __init__(self):
        self.test_loader = None
        self.val_loader = None
        self.train_loader = None

    @classmethod
    def register(cls):
        cls.builder_mapping[cls.name()] = cls
        logger.debug(f"Registering processor class  {cls.__name__} '{cls.name()}'")

    @classmethod
    def get(cls, name):
        if name not in cls.builder_mapping:
            raise ValueError(f"Unknown builder: {name}")
        return cls.builder_mapping[name]

    def build_dataloaders(
        self,
        x_path: str,
        y_path: str,
        train_ratio: float,
        val_ratio: float,
        batch_size: int,
    ):
        logger.info("Building dataloaders")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            x_path, y_path, train_ratio, val_ratio, batch_size
        )
        logger.info("Dataloaders built")
        return self

    def build_model(
        self,
        model_type: str,
        criterion_type: str,
        optimizer_type: str,
        scheduler_type: str,
        lr: float,
        device: torch.device,
        model_args: Optional[dict] = None,
        criterion_args: Optional[dict] = None,
        optimizer_args: Optional[dict] = None,
        scheduler_args: Optional[dict] = None,
    ):
        logger.info(
            f"Buiding model of type {model_type}, criterion {criterion_type}, optimizer {optimizer_type}, scheduler {scheduler_type}, lr {lr}"
        )
        model = get_model_class(model_type)(**model_args)
        criterion = get_criterion_class(criterion_type)(**criterion_args)
        optimizer = get_optimizer_class(optimizer_type)(
            model.parameters(), lr=lr, **optimizer_args
        )
        scheduler = get_scheduler_class(scheduler_type)(optimizer, **scheduler_args)
        self.model = PyTorchModel(model, criterion, optimizer, scheduler, device)

        logger.info("Adding l2 criterion")
        self.model.add_criteria("L2", L2Loss())

        logger.info("Adding linf criterion")
        self.model.add_criteria("Linf", LinfLoss())

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


def get_builder_class(name):
    return InverseBuilder.get(name)


package_dir = os.path.dirname(__file__)
import_modules(package_dir, "aidtep.inverse_problem")
