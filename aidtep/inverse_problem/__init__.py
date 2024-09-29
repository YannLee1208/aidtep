import os
from abc import ABC, abstractmethod
from typing import Optional

import torch
from loguru import logger

from aidtep.utils.common import Registry, import_modules


class InverseBuilder(Registry, ABC):
    builder_mapping = {}

    @classmethod
    def register(cls):
        cls.builder_mapping[cls.name()] = cls
        logger.debug(f"Registering processor class  {cls.__name__} '{cls.name()}'")

    @classmethod
    def get(cls, name):
        if name not in cls.builder_mapping:
            raise ValueError(f"Unknown builder: {name}")
        return cls.builder_mapping[name]

    @abstractmethod
    def build_dataloaders(
        self,
        x_path: str,
        y_path: str,
        train_ratio: float,
        val_ratio: float,
        batch_size: int,
    ):
        pass

    @abstractmethod
    def build_model(
        self,
        model_type: str,
        criterion_type: str,
        optimizer_type: str,
        scheduler_type: str,
        lr: float,
        device: torch.device,
        criterion_args: Optional[dict] = None,
        optimizer_args: Optional[dict] = None,
        scheduler_args: Optional[dict] = None,
    ):
        pass

    @abstractmethod
    def train(self, epochs: int, model_path: str):
        pass

    @abstractmethod
    def test(self, model_path: str):
        pass


def get_builder_class(name):
    return InverseBuilder.get(name)


package_dir = os.path.dirname(__file__)
import_modules(package_dir, "aidtep.inverse_problem")
