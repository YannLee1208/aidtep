import unittest
import numpy as np
from loguru import logger

from aidtep.data_process.interpolation import voronoi_tessellation


class TestVoronoiInterpolation(unittest.TestCase):

    def setUp(self):
        self.test_cases = [
            {
                "name": "normal_case",
                "input": {
                    "observations": np.array([[1, 2, 3], [4, 5, 6]]),
                    "sensor_position_mask": np.array([
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]
                    ])
                },
                "expected_output": np.array([
                    [
                        [1, 2, 2],
                        [1, 3, 3],
                        [1, 3, 3]
                    ],
                    [
                        [4, 5, 5],
                        [4, 6, 6],
                        [4, 6, 6]
                    ]
                ]),
                "error": None
            },
            {
                "name": "single_sensor",
                "input": {
                    "observations": np.array([[5]]),
                    "sensor_position_mask": np.array([
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]
                    ])
                },
                "expected_output": np.array([
                    [
                        [5, 5, 5],
                        [5, 5, 5],
                        [5, 5, 5]
                    ]
                ]),
                "error": None
            },
            {
                "name": "multiple_sensors",
                "input": {
                    "observations": np.array([
                        [1, 2, 3, 4],
                        [5, 6, 7, 8]
                    ]),
                    "sensor_position_mask": np.array([
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0]
                    ])
                },
                "expected_output": np.array([
                    [
                        [1, 2, 4],
                        [4, 2, 2],
                        [3, 4, 2],
                        [3, 3, 3]
                    ],
                    [
                        [5, 6, 8],
                        [8, 6, 6],
                        [7, 8, 6],
                        [7, 7, 7]
                    ]
                ]),
                "error": None
            },
            {
                "name": "no_sensors",
                "input": {
                    "observations": np.array([[1, 2], [3, 4]]),
                    "sensor_position_mask": np.zeros((3, 3))
                },
                "expected_output": None,
                "error": ValueError
            }
        ]

    def test_voronoi_interpolation(self):
        for case in self.test_cases:
            with self.subTest(case["name"]):
                if case["error"]:
                    with self.assertRaises(case["error"]):
                        voronoi_tessellation(**case["input"])
                else:
                    result = voronoi_tessellation(**case["input"])
                    logger.debug(result)
                    np.testing.assert_array_almost_equal(result, case["expected_output"])


if __name__ == "__main__":
    unittest.main()
