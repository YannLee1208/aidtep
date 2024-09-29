import unittest

import numpy as np

from aidtep.data_process.component.common import (
    normalize_2d_array,
    extract_observations,
    down_sample_3d_data,
)


class TestNormalize2DArray(unittest.TestCase):

    def setUp(self):
        self.test_cases = [
            {
                "name": "normal_case",
                "input": np.array([[1, 2], [3, 4]]),
                "expected_output": (
                    np.array([[0.25, 0.33333333], [0.75, 0.66666667]]),
                    np.array([4, 6]),
                ),
                "mock": None,
                "error": None,
            },
            {
                "name": "non_np_array",
                "input": [[1, 2], [3, 4]],
                "expected_output": None,
                "mock": None,
                "error": ValueError,
            },
            {
                "name": "non_2d_array",
                "input": np.array([1, 2, 3, 4]),
                "expected_output": None,
                "mock": None,
                "error": ValueError,
            },
        ]

    def test_normalize_2d_array(self):
        for case in self.test_cases:
            with self.subTest(case["name"]):
                if case["error"]:
                    with self.assertRaises(case["error"]):
                        normalize_2d_array(case["input"])
                else:
                    result = normalize_2d_array(case["input"])
                    np.testing.assert_array_almost_equal(
                        result[0], case["expected_output"][0]
                    )
                    np.testing.assert_array_almost_equal(
                        result[1], case["expected_output"][1]
                    )


class TestExtractObservations(unittest.TestCase):

    def setUp(self):
        self.test_extract_observations_cases = [
            {
                "name": "normal_2d_case",
                "input": {
                    "data": np.array(
                        [
                            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                        ]
                    ),
                    "mask": np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]]),
                },
                "expected_output": np.array([[2, 4, 6], [8, 6, 4]]),
                "error": None,
            },
            {
                "name": "empty_mask",
                "input": {
                    "data": np.array(
                        [
                            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                        ]
                    ),
                    "mask": np.zeros((3, 3)),
                },
                "expected_output": np.zeros((2, 0)),
                "error": None,
            },
            {
                "name": "single_sensor_mask",
                "input": {
                    "data": np.array(
                        [
                            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                        ]
                    ),
                    "mask": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                },
                "expected_output": np.array([[5], [5]]),
                "error": None,
            },
            {
                "name": "normal_3d_case",
                "input": {
                    "data": np.array(
                        [
                            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                        ]
                    ),
                    "mask": np.array(
                        [
                            [[0, 1, 0], [1, 0, 1], [0, 0, 0]],
                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        ]
                    ),
                },
                "expected_output": np.array([[2, 4, 6], [9, 5, 1]]),
                "error": None,
            },
            {
                "name": "different_sensors_each_sample",
                "input": {
                    "data": np.array(
                        [
                            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                        ]
                    ),
                    "mask": np.array(
                        [
                            [[0, 1, 0], [1, 0, 1], [0, 0, 0]],
                            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                        ]
                    ),
                },
                "expected_output": np.array([[2, 4, 6], [7, 5, 3]]),
                "error": None,
            },
            {
                "name": "empty_3d_masks",
                "input": {
                    "data": np.array(
                        [
                            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                        ]
                    ),
                    "mask": np.zeros((2, 3, 3)),
                },
                "expected_output": np.zeros((2, 0)),
                "error": None,
            },
            {
                "name": "1d_mask",
                "input": {
                    "data": np.array(
                        [
                            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        ]
                    ),
                    "mask": np.zeros((2,)),
                },
                "expected_output": np.zeros((2, 0)),
                "error": ValueError,
            },
            {
                "name": "3d_mask_shape_not_equal",
                "input": {
                    "data": np.array(
                        [
                            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                        ]
                    ),
                    "mask": np.zeros((2, 3, 1)),
                },
                "expected_output": np.zeros((2, 0)),
                "error": ValueError,
            },
        ]

    def test_extract_observations(self):
        for case in self.test_extract_observations_cases:
            with self.subTest(case["name"]):
                if case["error"]:
                    with self.assertRaises(case["error"]):
                        extract_observations(**case["input"])
                else:
                    result = extract_observations(**case["input"])
                    np.testing.assert_array_equal(result, case["expected_output"])


class TestDownSample3dData(unittest.TestCase):
    def setUp(self):
        self.test_down_sample_3d_data_cases = [
            {
                "name": "normal_case_mean",
                "input": {
                    "data": np.array(
                        [
                            [[1, 2, 3, 4], [5, 6, 7, 8]],
                            [[9, 10, 11, 12], [13, 14, 15, 16]],
                        ]
                    ),
                    "down_sample_factor": 2,
                    "down_sample_strategy": "mean",
                },
                "expected_output": np.array([[[3.5, 5.5]], [[11.5, 13.5]]]),
                "error": None,
            },
            {
                "name": "normal_case_max",
                "input": {
                    "data": np.array(
                        [
                            [[1, 2, 3, 4], [5, 6, 7, 8]],
                            [[9, 10, 11, 12], [13, 14, 15, 16]],
                        ]
                    ),
                    "down_sample_factor": 2,
                    "down_sample_strategy": "max",
                },
                "expected_output": np.array([[[6, 8]], [[14, 16]]]),
                "error": None,
            },
            {
                "name": "normal_case_min",
                "input": {
                    "data": np.array(
                        [
                            [[1, 2, 3, 4], [5, 6, 7, 8]],
                            [[9, 10, 11, 12], [13, 14, 15, 16]],
                        ]
                    ),
                    "down_sample_factor": 2,
                    "down_sample_strategy": "min",
                },
                "expected_output": np.array([[[1, 3]], [[9, 11]]]),
                "error": None,
            },
            {
                "name": "invalid_strategy",
                "input": {
                    "data": np.array(
                        [
                            [[1, 2, 3, 4], [5, 6, 7, 8]],
                            [[9, 10, 11, 12], [13, 14, 15, 16]],
                        ]
                    ),
                    "down_sample_factor": 2,
                    "down_sample_strategy": "median",
                },
                "expected_output": None,
                "error": ValueError,
            },
            {
                "name": "no_down_sampling",
                "input": {
                    "data": np.array(
                        [
                            [[1, 2, 3, 4], [5, 6, 7, 8]],
                            [[9, 10, 11, 12], [13, 14, 15, 16]],
                        ]
                    ),
                    "down_sample_factor": 1,
                    "down_sample_strategy": "mean",
                },
                "expected_output": np.array(
                    [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]
                ),
                "error": None,
            },
        ]

    def test_down_sample_3d_data(self):
        for case in self.test_down_sample_3d_data_cases:
            with self.subTest(case["name"]):
                if case["error"]:
                    with self.assertRaises(case["error"]):
                        down_sample_3d_data(**case["input"])
                else:
                    result = down_sample_3d_data(**case["input"])
                    np.testing.assert_array_equal(result, case["expected_output"])


if __name__ == "__main__":
    unittest.main()
