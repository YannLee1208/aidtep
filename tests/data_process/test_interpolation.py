import unittest

import numpy as np
from loguru import logger

from aidtep.data_process.component.interpolation import (
    get_interpolator_class,
    VoronoiInterpolator,
    VoronoiInterpolatorLinear,
)


class TestVoronoiInterpolation(unittest.TestCase):

    def setUp(self):
        self.test_get_interpolator_cases = [
            {
                "name": "valid_method_voronoi",
                "input": "voronoi",
                "expected_output": VoronoiInterpolator,
                "error": None,
            },
            {
                "name": "valid_method_voronoi_linear",
                "input": "voronoi_linear",
                "expected_output": VoronoiInterpolatorLinear,
                "error": None,
            },
            {
                "name": "invalid_method",
                "input": "invalid_method",
                "expected_output": None,
                "error": ValueError,
            },
        ]

        self.test_voronoi_cases = [
            {
                "name": "normal_case",
                "input": {
                    "observations": np.array([[1, 2, 3], [4, 5, 6]]),
                    "sensor_position_mask": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                },
                "expected_output": np.array(
                    [
                        [[1, 1, 1], [2, 1, 3], [2, 3, 3]],
                        [[4, 4, 4], [5, 4, 6], [5, 6, 6]],
                    ]
                ),
                "error": None,
            },
            {
                "name": "single_sensor",
                "input": {
                    "observations": np.array([[5]]),
                    "sensor_position_mask": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                },
                "expected_output": np.array([[[5, 5, 5], [5, 5, 5], [5, 5, 5]]]),
                "error": None,
            },
            {
                "name": "multiple_sensors",
                "input": {
                    "observations": np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
                    "sensor_position_mask": np.array(
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
                    ),
                },
                "expected_output": np.array(
                    [
                        [[1, 1, 2], [1, 2, 2], [4, 2, 3], [4, 4, 3]],
                        [[5, 5, 6], [5, 6, 6], [8, 6, 7], [8, 8, 7]],
                    ]
                ),
                "error": None,
            },
            {
                "name": "no_sensors",
                "input": {
                    "observations": np.array([[1, 2], [3, 4]]),
                    "sensor_position_mask": np.zeros((3, 3)),
                },
                "expected_output": None,
                "error": ValueError,
            },
        ]

        self.test_voronoi_linear_cases = [
            {
                "name": "linear_interpolation",
                "interpolator": VoronoiInterpolatorLinear(),
                "input": {
                    "observations": np.array([[1, 2, 3], [4, 5, 6]]),
                    "sensor_position_mask": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                },
                "expected_output": np.array(
                    [
                        [[np.nan, 1, np.nan], [2, 2, np.nan], [np.nan, np.nan, 3]],
                        [[np.nan, 4, np.nan], [5, 5, np.nan], [np.nan, np.nan, 6]],
                    ]
                ),
                "error": None,
            }
        ]

    def test_voronoi_interpolation(self):
        for case in self.test_voronoi_cases:
            with self.subTest(case["name"]):
                if case["error"]:
                    with self.assertRaises(case["error"]):
                        interpolator = VoronoiInterpolator()
                        interpolator.interpolate(**case["input"])
                else:
                    interpolator = VoronoiInterpolator()
                    result = interpolator.interpolate(**case["input"])
                    logger.debug(result)
                    np.testing.assert_array_almost_equal(
                        result, case["expected_output"]
                    )

    def test_voronoi_linear_interpolation(self):
        for case in self.test_voronoi_linear_cases:
            with self.subTest(case["name"]):
                if case["error"]:
                    with self.assertRaises(case["error"]):
                        interpolator = VoronoiInterpolatorLinear()
                        interpolator.interpolate(**case["input"])
                else:
                    interpolator = VoronoiInterpolatorLinear()
                    result = interpolator.interpolate(**case["input"])
                    logger.debug(result)
                    np.testing.assert_array_almost_equal(
                        result, case["expected_output"]
                    )

    def test_get_interpolator(self):
        for case in self.test_get_interpolator_cases:
            with self.subTest(case["name"]):
                if case["error"]:
                    with self.assertRaises(case["error"]):
                        get_interpolator_class(case["input"])()
                else:
                    interpolator = get_interpolator_class(case["input"])()
                    self.assertIsInstance(interpolator, case["expected_output"])


if __name__ == "__main__":
    unittest.main()
