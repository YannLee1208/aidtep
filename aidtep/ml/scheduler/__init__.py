import os

from abc import ABC, abstractmethod
from aidtep.utils.common import Registry, import_modules


class SchedulerRegistry(Registry, ABC):
    scheduler_mapping = {}

    @classmethod
    @abstractmethod
    def name(cls):
        pass

    @classmethod
    def register(cls):
        cls.scheduler_mapping[cls.name()] = cls

    @classmethod
    def get(cls, name):
        if name not in cls.scheduler_mapping:
            raise ValueError(f"Unknown scheduler type '{name}', choose from {cls.scheduler_mapping.keys()}")
        return cls.scheduler_mapping[name]


def get_scheduler_class(scheduler_type):
    scheduler = SchedulerRegistry.get(scheduler_type)
    return scheduler


package_dir = os.path.dirname(__file__)
import_modules(package_dir, 'aidtep.ml.scheduler')
