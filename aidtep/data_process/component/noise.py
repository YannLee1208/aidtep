import numpy as np


def add_noise(data: np.ndarray, noise_ratio: float) -> np.ndarray:
    """
    Add Gaussian noise to each sample in the data.
    :param data: np.ndarray, shape (n_sample, obs) or (n_sample, x_shape, y_shape)
    :param noise_ratio: float, standard deviation of the Gaussian noise relative to the data
    :return: np.ndarray, data with added noise
    """
    if noise_ratio < 0:
        raise ValueError("noise_ratio must be non-negative.")

    if noise_ratio == 0:
        return data

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array.")

    # generate Gaussian noise
    noise = np.random.normal(0, 1, data.shape)

    # add noise to data
    noisy_data = data * (1 + noise_ratio * noise)

    return noisy_data