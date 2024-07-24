import os
from typing import Literal

import numpy as np

from aidtep.data_process.common import down_sample_3d_data
from aidtep.utils import constants


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


def main_process(data_path: str, data_type: str = "float16", down_sample_factor: int = 1, down_sample_strategy: Literal["min", "max", "mean"] = "mean") -> np.ndarray:
    """
    Main process of IAEA raw data.
    :param data_path: IAEA raw data path
    :param data_type: data type, default: float16
    :param down_sample_factor: down sample factor, default: 1
    :param down_sample_strategy: down sample strategy, default: "mean"
    :return: IAEA raw data
    """
    data = load_IAEA_raw_data(data_path=data_path, data_type=data_type)
    data = down_sample_3d_data(data, down_sample_factor=down_sample_factor, down_sample_strategy=down_sample_strategy)
    return data
