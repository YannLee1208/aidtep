import os
from abc import ABC, abstractmethod

import torch

from aidtep.utils.common import Registry, import_modules


class BasisExtractorRegistry(Registry, ABC):
    basis_extractor_mapping = {}

    @classmethod
    def register(cls):
        cls.basis_extractor_mapping[cls.name()] = cls

    @classmethod
    def get(cls, name):
        if name not in cls.basis_extractor_mapping:
            raise ValueError(
                f"Unknown basis extractor type '{name}', choose from {cls.basis_extractor_mapping.keys()}"
            )
        return cls.basis_extractor_mapping[name]

    def __init__(self, device: torch.device):
        self.device = device
        self.basis = None
        self.basis_importance = None

    @abstractmethod
    def extract(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_basis(self) -> torch.Tensor:
        """
        Get the basis extracted from the full matrix
        :return: basis matrix, shape (n_basis, n_features)
        """
        pass

    @abstractmethod
    def get_basis_importance(self) -> torch.Tensor:
        """
        Get the importance of each basis
        :return: importance of each basis, shape (n_basis,)
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save the basis extractor to a file
        :param path: path to save the basis extractor
        """
        pass


def get_basis_extractor(basis_extractor_type: str):
    model = BasisExtractorRegistry.get(basis_extractor_type)
    return model


package_dir = os.path.dirname(__file__)
import_modules(package_dir, "aidtep.extract_basis")
