from abc import abstractmethod, ABC
import numpy as np
from loguru import logger
from scipy.interpolate import griddata

from aidtep.utils.common import Registry


class Interpolator(Registry, ABC):
    interpolator_mapping = {}

    @classmethod
    def register(cls):
        cls.interpolator_mapping[cls.name()] = cls
        logger.info(f"Registering interpolation class  {cls.__name__} '{cls.name()}'")

    @classmethod
    def get(cls, name):
        if name not in cls.interpolator_mapping:
            raise ValueError(f"Unknown interpolation method: {name}")
        return cls.interpolator_mapping[name]

    @abstractmethod
    def interpolate(self, observations: np.ndarray, sensor_position_mask: np.ndarray) -> np.ndarray:
        pass


class VoronoiInterpolator(Interpolator):
    @classmethod
    def name(cls):
        return "voronoi"

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
        x_grid = np.linspace(0, x_shape - 1, x_shape)
        y_grid = np.linspace(0, y_shape - 1, y_shape)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_points = np.array([X.flatten(), Y.flatten()]).T

        interpolated_data = np.zeros((n_sample, x_shape, y_shape))

        for i in range(n_sample):
            # voronoi tessellation
            Z = griddata(sensor_positions, observations[i], grid_points, method='nearest').reshape((y_shape, x_shape)).T
            interpolated_data[i] = Z
        return interpolated_data


class VoronoiInterpolatorLinear(Interpolator):
    @classmethod
    def name(cls):
        return "voronoi_linear"

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
        x_grid = np.linspace(0, x_shape - 1, x_shape)
        y_grid = np.linspace(0, y_shape - 1, y_shape)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_points = np.array([X.flatten(), Y.flatten()]).T

        interpolated_data = np.zeros((n_sample, x_shape, y_shape))

        for i in range(n_sample):
            # voronoi tessellation
            Z = griddata(sensor_positions, observations[i], grid_points, method='linear').reshape((y_shape, x_shape)).T
            interpolated_data[i] = Z
        return interpolated_data


def get_interpolator_class(interpolation_method):
    interpolator = Interpolator.get(interpolation_method)
    return interpolator

