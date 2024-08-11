from typing import Literal

import numpy as np
from loguru import logger

from aidtep.data_process.processor import DataProcessor
from aidtep.utils import constants
from aidtep.utils.file import check_file_exist


class CorcasimDataProcessor(DataProcessor):
    @classmethod
    def name(cls):
        return "Corcasim"

    def __init__(self, true_data_path: str, obs_output_path: str, interpolation_output_path: str):
        super().__init__(true_data_path, obs_output_path, interpolation_output_path)
        self.position_mask = None
        self.observation = None
        self.raw_data = None

    def load_raw_data(self, data_type: str, data_path_10000: str, data_path_8480: str):
        if not check_file_exist(data_path_10000):
            raise FileNotFoundError(f"{data_path_10000} not found")
        if not check_file_exist(data_path_8480):
            raise FileNotFoundError(f"{data_path_8480} not found")

        data_10000 = np.loadtxt(data_path_10000).reshape(-1, *constants.CORCASIM_DATA_SHAPE)
        data_8480 = np.loadtxt(data_path_8480).reshape(-1, *constants.CORCASIM_DATA_SHAPE)
        self.raw_data = np.concatenate([data_10000, data_8480], axis=0)
        logger.debug(f"Raw data shape: {self.raw_data.shape}")

    def down_sample_raw_data(self, down_sample_factor: int, down_sample_strategy: Literal["min", "max", "mean"]):
        pass

    def save_raw_data(self, *args, **kwargs):
        pass

    def get_observation(self, *args, **kwargs):
        pass

    def interpolate(self, *args, **kwargs):
        pass
