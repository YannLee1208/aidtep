import unittest

import numpy as np

from aidtep.data_process.component.sensor_position import (
    generate_2d_eye_mask,
    generate_2d_uniform_mask,
    generate_2d_specific_mask,
)


class TestGenerateMasks(unittest.TestCase):

    def setUp(self):
        self.test_generate_2d_eye_mask_cases = [
            {
                "name": "3x3_eye_mask",
                "input": 3,
                "expected_output": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                "error": None,
            },
            {
                "name": "1x1_eye_mask",
                "input": 1,
                "expected_output": np.array([[1]]),
                "error": None,
            },
        ]

        self.test_generate_2d_uniform_mask_cases = [
            {
                "name": "4x4_uniform_mask_2_sensors",
                "input": (4, 4, 2, 2),
                "expected_output": np.array(
                    [[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]]
                ),
                "error": None,
            },
            {
                "name": "3x3_uniform_mask_3_sensors",
                "input": (3, 3, 3, 3),
                "expected_output": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                "error": None,
            },
            {
                "name": "5x3_uniform_mask_2x2_sensors",
                "input": (5, 3, 2, 2),
                "expected_output": np.array(
                    [[1, 0, 1], [0, 0, 0], [0, 0, 0], [1, 0, 1], [0, 0, 0]]
                ),
                "error": None,
            },
            {
                "name": "5x5_uniform_mask_2x3_sensors",
                "input": (5, 5, 2, 3),
                "expected_output": np.array(
                    [
                        [1, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                "error": None,
            },
            {
                "name": "invalid_x_shape",
                "input": (-1, 4, 2, 2),
                "expected_output": None,
                "error": ValueError,
            },
        ]

    def test_generate_2d_eye_mask(self):
        for case in self.test_generate_2d_eye_mask_cases:
            with self.subTest(case["name"]):
                if case["error"]:
                    with self.assertRaises(case["error"]):
                        generate_2d_eye_mask(case["input"])
                else:
                    result = generate_2d_eye_mask(case["input"])
                    np.testing.assert_array_equal(result, case["expected_output"])

    def test_generate_2d_uniform_mask(self):
        for case in self.test_generate_2d_uniform_mask_cases:
            with self.subTest(case["name"]):
                if case["error"]:
                    with self.assertRaises(case["error"]):
                        generate_2d_uniform_mask(*case["input"])
                else:
                    result = generate_2d_uniform_mask(*case["input"])
                    np.testing.assert_array_equal(result, case["expected_output"])


class TestGenerate2DSpecificMask(unittest.TestCase):

    def setUp(self):
        self.test_cases = [
            {
                "name": "normal_case",
                "input": (5, 5, [1, 3], [2, 4]),
                "expected_output": np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                "error": None,
            },
            {
                "name": "out_of_bounds_x",
                "input": (5, 5, [1, 6], [2, 4]),
                "expected_output": None,
                "error": ValueError,
            },
            {
                "name": "out_of_bounds_y",
                "input": (5, 5, [1, 3], [2, 6]),
                "expected_output": None,
                "error": ValueError,
            },
            {
                "name": "empty_positions",
                "input": (5, 5, [], []),
                "expected_output": np.zeros((5, 5), dtype=int),
                "error": None,
            },
            {
                "name": "single_position",
                "input": (3, 3, [1], [1]),
                "expected_output": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                "error": None,
            },
        ]

    def test_generate_2d_specific_mask(self):
        for case in self.test_cases:
            with self.subTest(case["name"]):
                if case["error"]:
                    with self.assertRaises(case["error"]):
                        generate_2d_specific_mask(*case["input"])
                else:
                    result = generate_2d_specific_mask(*case["input"])
                    np.testing.assert_array_equal(result, case["expected_output"])


if __name__ == "__main__":
    unittest.main()
