from typing import Tuple
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


