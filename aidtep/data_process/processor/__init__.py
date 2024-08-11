import os

from abc import ABC, abstractmethod
from typing import Literal

from loguru import logger

from aidtep.utils.common import Registry, import_modules


class DataProcessor(Registry):
    processor_mapping = {}

    def __init__(self, true_data_path: str, obs_output_path: str, interpolation_output_path: str):
        super().__init__()
        self.true_data_path = true_data_path
        self.obs_output_path = obs_output_path
        self.interpolation_output_path = interpolation_output_path

    @classmethod
    def register(cls):
        cls.processor_mapping[cls.name()] = cls
        logger.debug(f"Registering processor class  {cls.__name__} '{cls.name()}'")

    @classmethod
    def get(cls, name):
        if name not in cls.processor_mapping:
            raise ValueError(f"Unknown processor: {name}")
        return cls.processor_mapping[name]

    @abstractmethod
    def load_raw_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def down_sample_raw_data(self, down_sample_factor: int, down_sample_strategy: Literal["min", "max", "mean"]):
        pass

    @abstractmethod
    def save_raw_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_observation(self, *args, **kwargs):
        pass

    @abstractmethod
    def interpolate(self, *args, **kwargs):
        pass


def get_processor_class(name):
    return DataProcessor.get(name)


package_dir = os.path.dirname(__file__)
import_modules(package_dir, 'aidtep.data_process.processor')