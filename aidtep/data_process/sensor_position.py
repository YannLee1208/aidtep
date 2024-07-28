import numpy as np


def generate_2d_eye_mask(n: int) -> np.ndarray:
    """
    Generate a 2D eye mask of size n x n
    :param n: int
    :return: np.ndarray
    """
    return np.eye(n)


def generate_2d_uniform_mask(x_shape: int, y_shape: int, x_sensor_number: int, y_sensor_number: int) -> np.ndarray:
    """
    Generate a 2D uniform mask of size x_shape x y_shape with x_sensor_number sensors in x and y_sensor_number sensors in y
    :param x_shape: int, x dimension of the mask
    :param y_shape: int, y dimension of the mask
    :param x_sensor_number: int, number of sensors in x
    :param y_sensor_number: int, number of sensors in y
    :return: np.ndarray, 2D uniform mask
    """
    x_step = (x_shape + 1) // x_sensor_number
    y_step = (y_shape + 1) // y_sensor_number
    mask = np.zeros((x_shape, y_shape), dtype=int)
    for i in range(x_sensor_number):
        for j in range(y_sensor_number):
            mask[i * x_step, j * y_step] = 1
    return mask


def generate_2d_specific_mask(x_shape: int, y_shape: int, x_mask_position: list, y_mask_position: list) -> np.ndarray:
    """
    Generate a 2D mask of size x_shape x y_shape with masks at specific positions.
    :param x_shape: int, x dimension of the mask
    :param y_shape: int, y dimension of the mask
    :param x_mask_position: list of int, positions in x where mask should be placed
    :param y_mask_position: list of int, positions in y where mask should be placed
    :return: np.ndarray, 2D mask with specific positions
    """
    mask = np.zeros((x_shape, y_shape), dtype=int)
    for x in x_mask_position:
        for y in y_mask_position:
            if 0 <= x < x_shape and 0 <= y < y_shape:
                mask[x, y] = 1
            else:
                raise ValueError(f"Invalid mask position ({x}, {y})")
    return mask


