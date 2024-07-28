import math
from typing import Tuple, Literal
import numpy as np


def normalize_2d_array(arr: np.array) -> Tuple[np.array, np.array]:
    """
    Normalize a 2D array to have each column sum to 1. Returns a tuple of the normalized array and the sum of each column.
    param arr: 2D array to normalize
    return: Tuple of the normalized array and the sum of each column
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if len(arr.shape) != 2:
        raise ValueError("Input must be a 2D array")

    col_sums = arr.sum(axis=0)
    normalized_arr = arr / col_sums
    return normalized_arr, col_sums


def down_sample_3d_data(data: np.ndarray, down_sample_factor: int, down_sample_strategy: Literal["min", "max", "mean"]) -> np.ndarray:
    """
    Down sample a 3D array by taking the min, max, or mean of each down sampled block.
    :param data: 3D array to down sample. (n_samples, height, width)
    :param down_sample_factor: Factor to down sample by
    :param down_sample_strategy: Strategy to use for down sampling. One of "min", "max", or "mean"
    :return: Down sampled 3D array
    """
    if down_sample_factor == 1:
        return data # no need to down sample

    # 1. calculate the new shape of the down sampled data
    new_shape = (data.shape[0], math.ceil(data.shape[1] / down_sample_factor),
                 math.ceil(data.shape[2] / down_sample_factor))

    # 2. create an empty array to store the down sampled data
    down_sampled_data = np.empty(new_shape)

    # 3. select the down sample strategy
    if down_sample_strategy == "max":
        f = np.max
    elif down_sample_strategy == "min":
        f = np.min
    elif down_sample_strategy == "mean":
        f = np.mean
    else:
        raise ValueError("Invalid down sample strategy: {}".format(down_sample_strategy))

    # 4. down sample the data
    for i in range(new_shape[1]):
        for j in range(new_shape[2]):
            down_sampled_data[:, i, j] = f(data[:,
                                          i * down_sample_factor: min((i + 1) * down_sample_factor,
                                                                       data.shape[1]),
                                          j * down_sample_factor: min((j + 1) * down_sample_factor,
                                                                       data.shape[2])
                                          ], axis=(1, 2))
    return down_sampled_data


def extract_observations(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract observations from the data using the provided mask.
    Handles both 2D and 3D masks.
    :param data: np.ndarray, shape (n, x_shape, y_shape), original data
    :param mask: np.ndarray, shape (x_shape, y_shape) or (n, x_shape, y_shape), mask matrix with sensor positions
    :return: np.ndarray, shape (n, sensor_count), extracted observations
    """
    n, x_shape, y_shape = data.shape

    if mask.ndim == 2:
        sensor_positions = np.argwhere(mask == 1)
        x_positions, y_positions = sensor_positions[:, 0], sensor_positions[:, 1]
        observations = data[:, x_positions, y_positions]
    elif mask.ndim == 3:
        if mask.shape != data.shape:
            raise ValueError("Mask shape must match data shape in 3D case")
        # use the first mask to determine the number of sensors
        template_positions = np.argwhere(mask[0] == 1)
        sensor_count = len(template_positions)

        # create an empty array to store the observations
        observations = np.zeros((n, sensor_count))

        for i in range(n):
            sensor_positions = np.argwhere(mask[i] == 1)
            x_positions, y_positions = sensor_positions[:, 0], sensor_positions[:, 1]
            observations[i, :] = data[i, x_positions, y_positions]
    else:
        raise ValueError("Mask must be either 2D or 3D")

    return observations

