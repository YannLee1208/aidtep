from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import griddata


class Interpolator(ABC):
    @abstractmethod
    def interpolate(self, observations: np.ndarray, sensor_position_mask: np.ndarray) -> np.ndarray:
        pass


class VoronoiInterpolator(Interpolator):
    def interpolate(self, observations: np.ndarray, sensor_position_mask: np.ndarray) -> np.ndarray:
        """
        Perform Voronoi interpolation on the observations.
        :param observations: np.ndarray, shape (n_sample, sensor_count), observation values
        :param sensor_position_mask: np.ndarray, shape (x_shape, y_shape), mask with sensor positions
        :return: np.ndarray, shape (n_sample, x_shape, y_shape), interpolated data
        """
        n_sample, sensor_count = observations.shape
        x_shape, y_shape = sensor_position_mask.shape

        # get sensor positions
        sensor_positions = np.argwhere(sensor_position_mask == 1)

        # get grid points
        x_grid = np.linspace(0, x_shape, x_shape)
        y_grid = np.linspace(0, y_shape, y_shape)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_points = np.array([X.flatten(), Y.flatten()]).T

        interpolated_data = np.zeros((n_sample, x_shape, y_shape))

        for i in range(n_sample):
            # voronoi tessellation
            Z = griddata(sensor_positions, observations[i], grid_points, method='nearest').reshape(x_shape, y_shape)
            interpolated_data[i] = Z
        return interpolated_data


class VoronoiInterpolatorLinear(Interpolator):
    def interpolate(self, observations: np.ndarray, sensor_position_mask: np.ndarray) -> np.ndarray:
        """
        Perform Voronoi interpolation on the observations.
        :param observations: np.ndarray, shape (n_sample, sensor_count), observation values
        :param sensor_position_mask: np.ndarray, shape (x_shape, y_shape), mask with sensor positions
        :return: np.ndarray, shape (n_sample, x_shape, y_shape), interpolated data
        """
        n_sample, sensor_count = observations.shape
        x_shape, y_shape = sensor_position_mask.shape

        # get sensor positions
        sensor_positions = np.argwhere(sensor_position_mask == 1)

        # get grid points
        x_grid = np.linspace(0, x_shape, x_shape)
        y_grid = np.linspace(0, y_shape, y_shape)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_points = np.array([X.flatten(), Y.flatten()]).T

        interpolated_data = np.zeros((n_sample, x_shape, y_shape))

        for i in range(n_sample):
            # voronoi tessellation
            Z = griddata(sensor_positions, observations[i], grid_points, method='linear').reshape(x_shape, y_shape)
            interpolated_data[i] = Z
        return interpolated_data


INTERPOLATION_MAPPING = {
    "voronoi": VoronoiInterpolator,
    "voronoi_linear": VoronoiInterpolatorLinear
}


def get_interpolator(interpolation_method):
    interpolator_class = INTERPOLATION_MAPPING.get(interpolation_method)
    if not interpolator_class:
        raise ValueError(f"Unknown interpolation method '{interpolation_method}'")
    return interpolator_class()
