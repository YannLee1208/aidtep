import numpy as np
import torch
from loguru import logger

from aidtep.extract_basis import BasisExtractorRegistry


class SVD(BasisExtractorRegistry):
    @classmethod
    def name(cls):
        return "SVD"

    def __init__(self, device: torch.device, base_number: int):
        """
        :param base_number: the number of basis to extract
        :param device: the device to run the SVD
        """
        super(SVD, self).__init__(device)
        self.base_number = base_number

    def extract(self, full_matrix: torch.Tensor):
        if isinstance(full_matrix, np.ndarray):
            full_matrix = torch.from_numpy(full_matrix)
        full_matrix = full_matrix.to(self.device)
        _, S, V = torch.linalg.svd(full_matrix, full_matrices=False)

        S = S[: self.base_number] / torch.sum(S[: self.base_number])

        self.basis = V[: self.base_number, :]
        self.basis_importance = S
        logger.info(f"base number = {self.base_number}, fij = {self.basis_importance}")

    def get_basis(self) -> torch.Tensor:
        if self.basis is None:
            raise ValueError("Please call extract() before get basis")
        return self.basis

    def get_basis_importance(self) -> torch.Tensor:
        if self.basis_importance is None:
            raise ValueError("Please call extract() before get basis importance")
        return self.basis_importance

    def save(self, path: str):
        """
        Save the basis and basis importance to the path
        :param path: the path to save the basis and basis importance
        :return: None
        """
        basis = self.basis.cpu().detach().numpy()
        basis_importance = self.basis_importance.cpu().detach().numpy()
        np.savez(path, basis=basis, basis_importance=basis_importance)
