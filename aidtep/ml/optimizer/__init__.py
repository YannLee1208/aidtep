import os

from abc import ABC, abstractmethod
from aidtep.utils.common import Registry, import_modules


class OptimizerRegistry(Registry, ABC):
    optimizer_mapping = {}

    @classmethod
    @abstractmethod
    def name(cls):
        pass

    @classmethod
    def register(cls):
        cls.optimizer_mapping[cls.name()] = cls

    @classmethod
    def get(cls, name):
        if name not in cls.optimizer_mapping:
            raise ValueError(f"Unknown optimizer type '{name}', choose from {cls.optimizer_mapping.keys()}")
        return cls.optimizer_mapping[name]


def get_optimizer_class(optimizer_type: str, **kwargs):
    optimizer = OptimizerRegistry.get(optimizer_type)
    return optimizer


package_dir = os.path.dirname(__file__)
import_modules(package_dir, 'aidtep.ml.optimizer')
