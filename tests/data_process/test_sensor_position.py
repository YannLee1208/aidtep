import unittest

import numpy as np

from aidtep.data_process.sensor_position import generate_2d_eye_mask, generate_2d_uniform_mask


class TestGenerateMasks(unittest.TestCase):

    def setUp(self):
        self.test_generate_2d_eye_mask_cases = [
            {
                "name": "3x3_eye_mask",
                "input": 3,
                "expected_output": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                "error": None
            },
            {
                "name": "1x1_eye_mask",
                "input": 1,
                "expected_output": np.array([[1]]),
                "error": None
            }
        ]

        self.test_generate_2d_uniform_mask_cases = [
            {
                "name": "4x4_uniform_mask_2_sensors",
                "input": (4, 4, 2, 2),
                "expected_output": np.array([[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]]),
                "error": None
            },
            {
                "name": "3x3_uniform_mask_3_sensors",
                "input": (3, 3, 3, 3),
                "expected_output": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                "error": None
            },
            {
                "name": "5x3_uniform_mask_2x2_sensors",
                "input": (5, 3, 2, 2),
                "expected_output": np.array([[1, 0, 1], [0, 0, 0], [0, 0, 0], [1, 0, 1], [0, 0, 0]]),
                "error": None
            },
            {
                "name": "5x5_uniform_mask_2x3_sensors",
                "input": (5, 5, 2, 3),
                "expected_output": np.array(
                    [[1, 0, 1, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 1, 0, 1], [0, 0, 0, 0, 0]]),
                "error": None
            },
            {
                "name": "invalid_x_shape",
                "input": (-1, 4, 2, 2),
                "expected_output": None,
                "error": ValueError
            }
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


if __name__ == "__main__":
    unittest.main()