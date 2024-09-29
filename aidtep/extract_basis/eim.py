import numpy as np
import torch
from loguru import logger

from aidtep.extract_basis import BasisExtractorRegistry


class BaseEIM(BasisExtractorRegistry):

    def __init__(self, base_number: int, device: torch.device, measure_matrix: np.ndarray):
        """
        Args:
            base_number (int): number of bases
            device (torch.device): device to run the algorithm
            measure_matrix (np.ndarray): measurement matrix, shape is (sensor_count, full_count)
        """
        self.base_number = base_number
        self.device = device

        self._count = 0
        self._residual = None
        self._obs_residual = None
        self._X = None
        self._Q = None
        self._L = torch.from_numpy(measure_matrix).to(self.device)
        self.errors = []

    @classmethod
    def name(cls):
        return "EIM"

    def get_basis(self) -> torch.Tensor:
        pass

    def get_basis_importance(self) -> torch.Tensor:
        pass

    def extract(self, full_matrix: torch.Tensor, obs_matrix: torch.Tensor):
        """
        :param full_matrix: shape is (sample_count, full_count)
        :param obs_matrix: shape is (sample_count, sensor_count)
        """
        full_matrix = full_matrix.T.to(self.device)
        obs_matrix = obs_matrix.T.to(self.device)

        # shape is (full_count, sample_count)
        interpolation = torch.zeros_like(full_matrix, device=self.device, dtype=obs_matrix.dtype)
        # shape is (full_count, base_number)
        self._Q = torch.zeros([full_matrix.shape[0], self.base_number], device=self.device, dtype=obs_matrix.dtype)
        # shape is (base_number)
        self._X = torch.zeros([self.base_number], device=self.device, dtype=torch.long)
        self._count = 0

        while self._count < self.base_number:
            self._residual = (full_matrix - interpolation)
            self._obs_residual = self._get_observation(self._residual)

            mu = self._find_mu()
            x = self._find_x(mu)
            qi = self._residual[:, mu] / self._obs_residual[x, mu]

            fij = torch.abs(self._obs_residual[x, mu]).item()

            # self.logger.log(f"count = {self._count}, fij = {fij}")
            if fij < 1e-12:
                break
            if self._count > 1 and fij / self.errors[-1] > 100:
                break

            self.errors.append(fij)

            self._X[self._count] = x
            self._Q[:, self._count] = qi

            self._count += 1
            interpolation = self._update_interpolation(obs_matrix, self._count)

        logger.info(f"count = {self._count}, fij = {self.errors}")

    def _find_mu(self) -> torch.Tensor:
        max_index = torch.argmax(torch.abs(self._obs_residual))  # 按照linf norm找出mu, x
        col_index = max_index % self._residual.shape[1]
        return col_index

    def _find_x(self, mu: torch.Tensor) -> torch.Tensor:
        max_index = torch.argmax(torch.abs(self._obs_residual))
        row_index = max_index // self._residual.shape[1]
        return row_index

    def _update_interpolation(self, obs_matrix: torch.Tensor, current_idx: int):
        Q = self._Q[:, :current_idx]  # shape: full_count * current_idx
        Q_obs = self._get_observation(Q)  # shape: sensor_count * current_idx

        M = Q_obs
        # print(torch.inverse( (M.T @ M).to(torch.float32)).to(torch.float16).shape)
        # print(M.shape)
        alpha = torch.inverse((M.T @ M)) @ M.T @ obs_matrix

        # M = Q_obs[self._X[:current_idx], :] # shape: base_number * current_idx
        # alpha = torch.inverse(M.to(torch.float32)).to(torch.float16).to(self.device) @ self.obs_matrix[self._X[:current_idx], :]

        return Q @ alpha

    def _get_observation(self, matrix: torch.Tensor) -> torch.Tensor:
        """ Calculate the observation matrix from the full matrix based on the linear measurement matrix L. Equation is y = L @ x.
        :param matrix (torch.Tensor): shape is (full_count, sample_count)
        :return: observation matrix, shape is (sensor_count, sample_count)
        """
        return self._L @ matrix
