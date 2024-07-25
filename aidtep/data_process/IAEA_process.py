import os
from typing import Literal

import numpy as np
from loguru import logger

from aidtep.data_process.common import down_sample_3d_data, extract_observations
from aidtep.data_process.sensor_position import generate_2d_specific_mask
from aidtep.utils import constants
from aidtep.utils.file import save_ndarray


def load_IAEA_raw_data(data_path: str, data_size: int = constants.IAEA_DATA_SIZE, data_shape: tuple = constants.IAEA_DATA_SHAPE,
                       data_type: str = "float16", ) -> np.ndarray:
    """
    Read IAEA raw data, reshape and convert data type. The data is stored in a text file, shape is (data_size * data_shape[0], data_shape[1]).
    :param data_path: IAEA raw data path
    :param data_size: Number of data
    :param data_shape: data shape, example: (171, 171)
    :param data_type: data type, default: float16
    :return: IAEA raw data
    """
    data = np.loadtxt(data_path)
    data = data.astype(data_type)
    data = data.reshape(data_size, *data_shape)
    return data


def main_process(data_path: str, data_type: str , down_sample_factor: int, down_sample_strategy: Literal["min", "max", "mean"],
                 x_sensor_position: list, y_sensor_position: list, random_range: int, noise_ratio: float,
                 obs_output_path: str) -> np.ndarray:
    """
    Main process of IAEA raw data.
    :param data_path: IAEA raw data path
    :param data_type: data type, default: float16
    :param down_sample_factor: down sample factor, default: 1
    :param down_sample_strategy: down sample strategy, default: "mean"
    :return: IAEA raw data
    """
    obs_output_path = obs_output_path.format(
        data_type=data_type, down_sample_factor=down_sample_factor, down_sample_strategy=down_sample_strategy,
        x_sensor_position=x_sensor_position, y_sensor_position=y_sensor_position, random_range=random_range,
        noise_ratio=noise_ratio
    )

    logger.info(f"obs_output_path: {obs_output_path}")

    data = load_IAEA_raw_data(data_path=data_path, data_type=data_type)
    logger.info(f"IAEA raw data shape: {data.shape}")
    data = down_sample_3d_data(data, down_sample_factor=down_sample_factor, down_sample_strategy=down_sample_strategy)
    logger.info(f"After down sample, IAEA raw data shape: {data.shape}")

    position_mask = generate_2d_specific_mask(data.shape[0], data.shape[1], x_sensor_position, y_sensor_position)
    observation = extract_observations(data, position_mask)
    logger.info(f"Observation shape: {observation.shape}")

    save_ndarray(file_path=obs_output_path, data=observation)
