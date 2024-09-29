import numpy as np


def generate_vibrated_masks(sensor_position_mask: np.ndarray, random_range: int, n: int) -> np.ndarray:
    """
    Generate n vibrated masks based on the given sensor_position_mask with random perturbations.
    :param sensor_position_mask: np.ndarray, 2D mask of sensor positions with 0/1 values
    :param random_range: int, radius of perturbation for sensor positions
    :param n: int, number of vibrated masks to generate
    :return: np.ndarray, shape (n, x_shape, y_shape), list of vibrated 2D masks
    """
    if random_range <= 0:
        raise ValueError("random_range must be a non-negative integer")

    x_shape, y_shape = sensor_position_mask.shape

    # retrieve sensor positions
    sensor_positions = np.argwhere(sensor_position_mask == 1)

    # generate random perturbations, shape (n, len(sensor_positions))
    perturbations_x = np.random.randint(-random_range, random_range + 1, (n, len(sensor_positions)))
    perturbations_y = np.random.randint(-random_range, random_range + 1, (n, len(sensor_positions)))

    # initialize vibrated masks
    vibrated_masks = np.zeros((n, x_shape, y_shape), dtype=int)

    for idx, (x, y) in enumerate(sensor_positions):
        new_x_positions = np.clip(x + perturbations_x[:, idx], 0, x_shape - 1)
        new_y_positions = np.clip(y + perturbations_y[:, idx], 0, y_shape - 1)

        for i in range(n):
            vibrated_masks[i, new_x_positions[i], new_y_positions[i]] = 1

    return vibrated_masks
