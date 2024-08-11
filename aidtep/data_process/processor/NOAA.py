from typing import Literal

import h5py
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from aidtep.data_process.common import down_sample_3d_data, extract_observations
from aidtep.data_process.interpolation import get_interpolator_class
from aidtep.data_process.noise import add_noise
from aidtep.data_process.processor import DataProcessor
from aidtep.data_process.sensor_position import generate_2d_specific_mask
from aidtep.data_process.vibration import generate_vibrated_masks
from aidtep.utils.file import check_file_exist, save_ndarray


class NOAADataProcessor(DataProcessor):
    @classmethod
    def name(cls):
        return "NOAA"

    def __init__(self, true_data_path: str, obs_output_path: str, interpolation_output_path: str):
        super().__init__(true_data_path, obs_output_path, interpolation_output_path)
        self.position_mask = None
        self.observation = None
        self.raw_data = None

    def load_raw_data(self, data_type: str, data_path: str):
        """
        Load IAEA raw data from txt file.
        :param data_type: str, data type of the raw data
        :param data_path: str, data path
        """
        if not check_file_exist(data_path):
            raise ValueError("Please provide all the input files.")
        f = h5py.File(data_path, 'r')
        lat = np.array(f['lat']).squeeze()
        lon = np.array(f['lon']).squeeze()
        height, width = lat.shape[0], lon.shape[0]
        raw_data = np.nan_to_num(np.array(f['sst'])).reshape((-1, width, height))
        self.raw_data = np.swapaxes(raw_data, 1, 2)[:, ::-1, :].astype(data_type)
        logger.debug(f"NOAA raw data shape: {self.raw_data.shape}")

    def down_sample_raw_data(self, down_sample_factor: int, down_sample_strategy: Literal["min", "max", "mean"]):
        """
        Down sample the raw data by a factor.
        For example, if the raw data shape is (100, 100, 100), and down sample
            factor is 2, the down sampled data shape will be (100, 50, 50).
        :param down_sample_factor: int, down sample factor
        :param down_sample_strategy: literal["min", "max", "mean"], down sample strategy
        """
        if self.raw_data is not None:
            self.raw_data = down_sample_3d_data(self.raw_data, down_sample_factor=down_sample_factor,
                                                down_sample_strategy=down_sample_strategy)
            logger.debug(f"After down sample, NOAA raw data shape: {self.raw_data.shape}")
        else:
            raise ValueError("Please load raw data first.")

    def save_raw_data(self, data_type: str, down_sample_factor: int,
                      down_sample_strategy: Literal["min", "max", "mean"] ):
        """
        Save the down sampled raw data to a file.
        :param down_sample_strategy: str, down sample strategy
        :param down_sample_factor: int, down sample factor
        :param data_type: str, data type of the raw data
        """
        logger.debug(f"Saving NOAA raw data to {self.true_data_path}")
        save_ndarray(self.true_data_path, self.raw_data)

    # TODO: support other type of sensor position
    def get_observation(self, x_sensor_position: list, y_sensor_position: list, random_range: int, noise_ratio: float,
                        ):
        """
        Generate observation data based on the sensor position.
        :param x_sensor_position: list, x position of the sensor
        :param y_sensor_position: list, y position of the sensor
        :param random_range: int, random range for vibration
        :param noise_ratio: float, noise ratio
        """
        if self.raw_data is None:
            self.raw_data = np.load(self.true_data_path)

        self.position_mask = generate_2d_specific_mask(self.raw_data.shape[1], self.raw_data.shape[2],
                                                       x_sensor_position,
                                                       y_sensor_position)
        if random_range > 0:
            self.position_mask = generate_vibrated_masks(self.position_mask, random_range=random_range,
                                                         n=self.raw_data.shape[0])

        self.observation = extract_observations(self.raw_data, self.position_mask)

        if noise_ratio > 0:
            self.observation = add_noise(self.observation, noise_ratio)

        logger.debug(f"Observation shape: {self.observation.shape}")
        save_ndarray(file_path=self.obs_output_path, data=self.observation)

            # plt.imshow(self.observation[0])
            # plt.savefig('observation.png')

    def interpolate(self, x_shape: int, y_shape: int, x_sensor_position: list, y_sensor_position: list,
                    method: str):
        """
        Interpolate the observation data based on the sensor position.
        :param x_shape: int, x shape of the interpolation
        :param y_shape: int, y shape of the interpolation
        :param x_sensor_position: list, x position of the sensor
        :param y_sensor_position: list, y position of the sensor
        :param method: str, tessellation type
        :return: Self
        """
        if self.observation is None:
            if not check_file_exist(self.obs_output_path):
                raise ValueError("Please generate observation first.")
            self.observation = np.load(self.obs_output_path)

        self.position_mask = generate_2d_specific_mask(x_shape, y_shape, x_sensor_position, y_sensor_position)
        interpolator = get_interpolator_class(method)()
        interpolation = interpolator.interpolate(self.observation, self.position_mask)
        logger.debug(f"After interpolation, tessellation shape: {interpolation.shape}")
        save_ndarray(file_path=self.interpolation_output_path, data=interpolation)
        logger.debug(f"Interpolation saved to {self.interpolation_output_path}")
        plt.imshow(interpolation[0])
        plt.savefig('interpolation.png')
