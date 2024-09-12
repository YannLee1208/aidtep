from aidtep.extract_basis import BasisExtractorRegistry

import torch
from loguru import logger


class SVD(BasisExtractorRegistry):
    @classmethod
    def name(cls):
        return "SVD"

    def __init__(self, base_number: int, device: torch.device):
        """
        Args:
            base_number (int): number of bases
            device (torch.device): device to run the algorithm
        """
        self.base_number = base_number
        self.device = device
        self.basis = None
        self.basis_importance = None

    def extract(self, full_matrix: torch.Tensor):
        full_matrix = full_matrix.to(self.device)
        _, S, V = torch.linalg.svd(full_matrix, full_matrices=False)

        S = S[:self.base_number] / torch.sum(S[:self.base_number])

        self.basis = V[:self.base_number, :]
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