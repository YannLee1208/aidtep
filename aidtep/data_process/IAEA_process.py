from aidtep.data_process.common import down_sample_3d_data, extract_observations
from aidtep.data_process.noise import add_noise
from aidtep.data_process.sensor_position import generate_2d_specific_mask
from aidtep.data_process.interpolation import voronoi_tessellation
from aidtep.data_process.vibration import generate_vibrated_masks
from aidtep.utils import constants
from aidtep.utils.file import save_ndarray, check_file_exist

import numpy as np
from typing import Literal
from loguru import logger


class IAEADataProcessBuilder:
    def __init__(self, obs_output_path: str, interpolation_output_path: str):
        self.obs_output_path = obs_output_path
        self.interpolation_output_path = interpolation_output_path
        self.data = None
        self.observation = None
        self.position_mask = None
        self.interpolation = None

    @staticmethod
    def load_raw_data(data_path: str, data_type: str) -> np.ndarray:
        """
        Load IAEA raw data from txt file.
        :param data_path: str, path to the txt file
        :param data_type: str, data type of the raw data
        :return: np.ndarray, IAEA raw data
        """
        if not check_file_exist(data_path):
            raise FileNotFoundError(f"File {data_path} not found.")
        data = np.loadtxt(data_path)
        data = data.astype(data_type)
        data = data.reshape(constants.IAEA_DATA_SIZE, *constants.IAEA_DATA_SHAPE)
        logger.debug(f"IAEA raw data shape: {data.shape}")
        return data

    def down_sample_raw_data(self, down_sample_factor: int, down_sample_strategy: Literal["min", "max", "mean"]):
        """
        Down sample the raw data by a factor.
        For example, if the raw data shape is (100, 100, 100), and down sample
            factor is 2, the down sampled data shape will be (100, 50, 50).
        :param down_sample_factor: int, down sample factor
        :param down_sample_strategy: literal["min", "max", "mean"], down sample strategy
        :return: Self
        """
        if self.data is not None:
            self.data = down_sample_3d_data(self.data, down_sample_factor=down_sample_factor,
                                            down_sample_strategy=down_sample_strategy)
            logger.debug(f"After down sample, IAEA raw data shape: {self.data.shape}")
        else:
            raise ValueError("Please load raw data first.")
        return self

    def get_true_date(self, phione_input_path: str, phione_output_path: str, phitwo_input_path: str,
                      phitwo_output_path: str,
                      power_input_path: str, power_output_path: str,
                      data_type: str,
                      down_sample_factor: int, down_sample_strategy: Literal["min", "max", "mean"]):
        phione = self.load_raw_data(phione_input_path, data_type)
        phione = down_sample_3d_data(phione, down_sample_factor, down_sample_strategy)
        save_ndarray(phione_output_path, phione)
        logger.debug(f"Save phione to {phione_output_path}, data shape: {phione.shape}")
        del phione

        phitwo = self.load_raw_data(phitwo_input_path, data_type)
        phitwo = down_sample_3d_data(phitwo, down_sample_factor, down_sample_strategy)
        save_ndarray(phitwo_output_path, phitwo)
        logger.debug(f"Save phitwo to {phitwo_output_path}, data shape: {phitwo.shape}")
        del phitwo

        power = self.load_raw_data(power_input_path, data_type)
        power = down_sample_3d_data(power, down_sample_factor, down_sample_strategy)
        save_ndarray(power_output_path, power)
        logger.debug(f"Save power to {power_output_path}, data shape: {power.shape}")
        del power

    # TODO: support other type of sensor position
    def get_observation(self, data_path: str, data_type: str,
                        down_sample_factor: int, down_sample_strategy: Literal["min", "max", "mean"],
                        x_sensor_position: list, y_sensor_position: list, random_range: int, noise_ratio: float):
        """
        Generate observation data based on the sensor position.
        :param x_sensor_position: list, x position of the sensor
        :param y_sensor_position: list, y position of the sensor
        :param random_range: int, random range for vibration
        :param noise_ratio: float, noise ratio
        :return: Self
        """
        self.data = self.load_raw_data(data_path, data_type)
        self.data = down_sample_3d_data(self.data, down_sample_factor=down_sample_factor, down_sample_strategy=down_sample_strategy)
        if self.data is not None:
            self.position_mask = generate_2d_specific_mask(self.data.shape[1], self.data.shape[2], x_sensor_position,
                                                           y_sensor_position)
            if random_range > 0:
                vibration_position_mask = generate_vibrated_masks(self.position_mask, random_range=random_range,
                                                                  n=self.data.shape[0])
                self.observation = extract_observations(self.data, vibration_position_mask)
            else:
                self.observation = extract_observations(self.data, self.position_mask)

            if noise_ratio > 0:
                self.observation = add_noise(self.observation, noise_ratio)

            logger.debug(f"Observation shape: {self.observation.shape}")
            save_ndarray(file_path=self.obs_output_path, data=self.observation)
        else:
            raise ValueError("Please load raw data first.")
        return self

    # TODO: Add tessellation type
    def interpolate(self, x_shape: int, y_shape: int, x_sensor_position: list, y_sensor_position: list,
                    tessellation_type: str):
        """
        Interpolate the observation data based on the sensor position.
        :param x_shape: int, x shape of the interpolation
        :param y_shape: int, y shape of the interpolation
        :param x_sensor_position: list, x position of the sensor
        :param y_sensor_position: list, y position of the sensor
        :param tessellation_type: str, tessellation type
        :return: Self
        """
        if self.observation is not None:
            if not check_file_exist(self.obs_output_path):
                raise ValueError("Please generate observation first.")
            self.observation = np.load(self.obs_output_path)

        self.position_mask = generate_2d_specific_mask(x_shape, y_shape, x_sensor_position, y_sensor_position)
        self.interpolation = voronoi_tessellation(self.observation, self.position_mask)
        logger.debug(f"After interpolation, tessellation shape: {self.interpolation.shape}")
        save_ndarray(file_path=self.interpolation_output_path, data=self.interpolation)
        logger.debug(f"Interpolation saved to {self.interpolation_output_path}")

        return self
