import importlib
import pkgutil
from abc import ABC, abstractmethod


class Registry(ABC):

    @classmethod
    @abstractmethod
    def name(cls):
        pass

    @classmethod
    @abstractmethod
    def register(cls):
        pass

    @classmethod
    @abstractmethod
    def get(cls, name):
        pass

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.register()


def import_modules(package_dir: str, module_name: str):
    """
    Import all modules in the package
    :param package_dir: directory of the package, usually __file__
    :param module_name: name of the package, ex: aidtep.ml.models
    """
    for (module_loader, name, ispkg) in pkgutil.iter_modules([package_dir]):
        if not ispkg and name != "__init__":
            importlib.import_module(f'{module_name}.{name}')
