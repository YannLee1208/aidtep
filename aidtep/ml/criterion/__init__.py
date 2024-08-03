import os

from abc import ABC, abstractmethod
from aidtep.utils.common import Registry, import_modules


class CriterionRegistry(Registry, ABC):
    criterion_mapping = {}

    @classmethod
    @abstractmethod
    def name(cls):
        pass

    @classmethod
    def register(cls):
        cls.criterion_mapping[cls.name()] = cls

    @classmethod
    def get(cls, name):
        if name not in cls.criterion_mapping:
            raise ValueError(f"Unknown criterion type '{name}', choose from {cls.criterion_mapping.keys()}")
        return cls.criterion_mapping[name]


def get_criterion_class(criterion_type):
    model = CriterionRegistry.get(criterion_type)
    return model


package_dir = os.path.dirname(__file__)
import_modules(package_dir, 'aidtep.ml.criterion')
