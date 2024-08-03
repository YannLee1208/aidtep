import os
from abc import ABC, abstractmethod

from aidtep.utils.common import Registry, import_modules


class ModelRegistry(Registry, ABC):
    model_mapping = {}

    @classmethod
    @abstractmethod
    def name(cls):
        pass
    @classmethod
    def register(cls):
        cls.model_mapping[cls.name()] = cls
        # logger.info(f"Registering model {cls.__name__} '{cls.name()}'")
        # logger.info(f"current mapping: {cls.model_mapping}")

    @classmethod
    def get(cls, name):
        # logger.info(f"getting model {name}")
        # logger.info(f"mapping: {cls.model_mapping}")
        if name not in cls.model_mapping:
            raise ValueError(f"Unknown model type '{name}', choose from {cls.model_mapping.keys()}")
        return cls.model_mapping[name]


def get_model_class(model_type):
    model = ModelRegistry.get(model_type)
    return model


package_dir = os.path.dirname(__file__)
import_modules(package_dir, 'aidtep.ml.models')

