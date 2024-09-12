from abc import abstractmethod

import numpy as np
import torch
from loguru import logger
from pathlib import Path

from aidtep.extract_basis import BasisExtractorRegistry


class BaseEIM(BasisExtractorRegistry):
    @classmethod
    def name(cls):
        return "EIM"

    def __init__(self, base_number: int, device: torch.device, measure_matrix: np.ndarray):
        """
        Args:
            base_number (int): number of bases
            device (torch.device): device to run the algorithm
            measure_matrix (np.ndarray): measurement matrix, shape is (sensor_count, full_count)
        """
        self._count = None
        self._residual = None
        self._obs_residual = None
        self._X = None
        self._Q = None
        self.base_number = base_number
        self.device = device
        self._L = torch.from_numpy(measure_matrix).to(self.device)
        self.errors = []

    def extract(self, obs_matrix: torch.Tensor, full_matrix: torch.Tensor):
        full_matrix = full_matrix.to(self.device)
        obs_matrix = obs_matrix.to(self.device)

        interpolation = torch.zeros_like(full_matrix, device=self.device, dtype=obs_matrix.dtype)
        self._Q = torch.zeros([full_matrix.shape[0], self.base_number], device=self.device, dtype=obs_matrix.dtype)
        self._X = torch.zeros([self.base_number], device=self.device, dtype=torch.long)

        self._count = 0

        while self._count < self.base_number:
            self._residual = (full_matrix - interpolation)
            self._obs_residual = self._get_observation(self._residual)

            mu = self._find_mu()
            x = self._find_x(mu)
            qi = self._cal_qi(mu, x)

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

    def _get_observation(self, matrix: torch.Tensor) -> torch.Tensor:
        """ Calculate the observation matrix from the full matrix based on the linear measurement matrix L. Equation is y = L @ x.

        Args:
            matrix (torch.Tensor): the full matrix

        Returns:
            torch.Tensor: the observation matrix
        """
        return self._L @ matrix

    @abstractmethod
    def _find_mu(self) -> torch.Tensor:
        """ Find the column index with the largest norm of the residual matrix.

        Returns:
            torch.Tensor: mu, the column index
        """
        pass

    @abstractmethod
    def _find_x(self, mu: torch.Tensor) -> torch.Tensor:
        """ Find the row index with the largest absolute value of the measurement residual.

        Args:
            mu (torch.Tensor): the column index

        Returns:
            torch.Tensor: x, the row index
        """
        pass

    def _cal_qi(self, mu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """ Calculate the ith column of the Q matrix.

        Args:
            mu (torch.Tensor): the column index
            x (torch.Tensor): the row index

        Returns:
            torch.Tensor: the ith column of the Q matrix
        """
        return self._residual[:, mu] / self._obs_residual[x, mu]

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

    def test(self, obs_matrix: torch.Tensor):
        obs_matrix = obs_matrix.to(self.device)

        # if has the attribute self._count
        if hasattr(self, "_count"):
            Q = self._Q[:, :self._count]  # shape: full_count * base_number
            X = self._X[:self._count]
        else:
            Q = self._Q
            X = self._X

        Q_obs = self._get_observation(Q)  # shape: sensor_count * base_number

        M = Q_obs
        alpha = torch.inverse((M.T @ M)) @ M.T @ obs_matrix

        # M = Q_obs[X, :] # shape: base_number * base_number
        # alpha = torch.inverse(M.to(torch.float32)).to(torch.float16).to(self.device) @ obs_matrix[X, :]
        # print(M)

        return Q @ alpha  # shape: full_count * parameter count

    def get_alpha(self, obs_matrix):
        obs_matrix = obs_matrix.to(self.device)
        if hasattr(self, "_count"):
            Q = self._Q[:, :self._count]  # shape: full_count * base_number
            X = self._X[:self._count]
        else:
            Q = self._Q
            X = self._X

        Q_obs = self._get_observation(Q)  # shape: sensor_count * base_number

        M = Q_obs
        alpha = torch.inverse((M.T @ M)) @ M.T @ obs_matrix
        return alpha.T

    def get_errors(self):
        return torch.Tensor(self.errors).to(torch.float64).to(self.device)

    def save(self, model_path):
        """
            store self._L, self._Q, self._X
        """
        if not Path(model_path).parent.exists():
            Path(model_path).parent.mkdir(parents=True)

        torch.save(
            [self._L, self._Q[:, :self._count], self._X[:self._count], self.errors], model_path
        )

    def load(self, model_path):
        """
            load self._L, self._Q, self._X
        """

        [self._L, self._Q, self._X, self.errors] = torch.load(model_path)
        self._L = self._L.to(self.device)
        self._Q = self._Q.to(self.device)
        self._X = self._X.to(self.device)
        self.errors = torch.Tensor(self.errors).to(self.device)


class EIM3(BaseEIM):
    @classmethod
    def name(cls):
        return "EIM3"

    def __init__(self, base_number: int, device: torch.device, measure_matrix: np.ndarray,  t=0):
        super().__init__(base_number, device, measure_matrix)
        self.t = t

    def _find_mu(self) -> torch.Tensor:
        max_index = torch.argmax(torch.abs(self._obs_residual))  # 按照linf norm找出mu, x
        col_index = max_index % self._residual.shape[1]
        return col_index

    def _find_x(self, mu: torch.Tensor) -> torch.Tensor:
        max_index = torch.argmax(torch.abs(self._obs_residual))
        row_index = max_index // self._residual.shape[1]
        return row_index

    def cal_M(self):
        if hasattr(self, "_count"):
            Q = self._Q[:, :self._count]  # shape: full_count * base_number
            X = self._X[:self._count]
        else:
            Q = self._Q
            X = self._X

        Q_obs = self._get_observation(Q)  # shape: sensor_count * base_number
        M = Q_obs
        # a = torch.diag(torch.Tensor(self.errors)).to(self.device) ** 2
        a = torch.diag(1 / torch.Tensor(self.errors)).to(self.device) ** 2
        return torch.inverse((M.T @ M + self.t * a)) @ M.T

    def test(self, obs_matrix: torch.Tensor):
        obs_matrix = obs_matrix.to(self.device)
        if hasattr(self, "_count"):
            Q = self._Q[:, :self._count]  # shape: full_count * base_number
            X = self._X[:self._count]
        else:
            Q = self._Q
            X = self._X

        # if has the attribute self._count

        alpha = self.cal_M() @ obs_matrix

        # M = Q_obs[X, :] # shape: base_number * base_number
        # alpha = torch.inverse(M.to(torch.float32)).to(torch.float16).to(self.device) @ obs_matrix[X, :]
        # print(M)

        return Q @ alpha  # shape: full_count * parameter count

    def get_alpha(self, obs_matrix):
        obs_matrix = obs_matrix.to(self.device)

        alpha = self.cal_M() @ obs_matrix
        return alpha.T

    def _update_interpolation(self, obs_matrix: torch.Tensor, current_idx: int):
        Q = self._Q[:, :current_idx]  # shape: full_count * current_idx
        alpha = self.cal_M() @ obs_matrix

        # M = Q_obs[self._X[:current_idx], :] # shape: base_number * current_idx
        # alpha = torch.inverse(M.to(torch.float32)).to(torch.float16).to(self.device) @ self.obs_matrix[self._X[:current_idx], :]

        return Q @ alpha


def position_mask_to_measure_matrix(position_mask: np.ndarray) -> np.ndarray:
    """
    Transform the position mask to the measurement matrix used in EIM.
    :param position_mask: np.ndarray, shape is (x_shape, y_shape)
    :return: measure_matrix: np.ndarray, shape is (sensor_count, full_count),
            full_count = x_shape * y_shape, sensor_count is the number of ones in the position mask
    """
    position_mask = position_mask.flatten()
    sensor_count = np.sum(position_mask)
    full_count = position_mask.size
    measure_matrix = np.zeros((sensor_count, full_count))
    sensor_idx = 0
    for i in range(full_count):
        if position_mask[i]:
            measure_matrix[sensor_idx, i] = 1
            sensor_idx += 1
    return measure_matrix


if __name__ == '__main__':
    from aidtep.data_process.sensor_position import generate_2d_specific_mask
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    x_shape = 180
    y_shape = 360
    x_sensor_position = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    y_sensor_position = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]

    position_mask = generate_2d_specific_mask(x_shape, y_shape, x_sensor_position, y_sensor_position)
    print(position_mask.shape)
    measure_matrix = position_mask_to_measure_matrix(position_mask)
    print(measure_matrix.shape)
    EIM = EIM3(20, torch.device("cuda:0"), measure_matrix)

    obs_path = "../../data/processed/NOAA/obs_float64_1_mean_vib_0_noise_0.0_[0, 20, 40, 60, 80, 100, 120, 140, 160]_[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340].npy"
    true_data_path = "../../data/processed/NOAA/NOAA_float64_1_mean.npy"

    obs_date = np.load(obs_path).T
    true_data = np.load(true_data_path).reshape(-1, x_shape * y_shape).T
    obs_date = torch.from_numpy(obs_date)
    true_data = torch.from_numpy(true_data)

    EIM.extract(obs_date, true_data)
    prediction = EIM.test(obs_date).cpu().detach().numpy().T
    print(prediction.shape)
    err = np.mean(np.abs(prediction - true_data.cpu().detach().numpy().T))
    print(err)
    plt.imshow(prediction[0].reshape(x_shape, y_shape))
    plt.savefig("test.png")


