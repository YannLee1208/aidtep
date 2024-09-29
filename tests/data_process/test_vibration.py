import unittest

import numpy as np

from aidtep.data_process.component.vibration import generate_vibrated_masks


class TestGenerateVibratedMasks(unittest.TestCase):

    def setUp(self):
        self.test_cases = [
            {
                "name": "normal_case",
                "input": {
                    "sensor_position_mask": np.array(
                        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
                    ),
                    "random_range": 1,
                    "n": 3,
                },
                "expected_output_shape": (3, 4, 4),
                "error": None,
            },
            {
                "name": "no_sensors",
                "input": {
                    "sensor_position_mask": np.zeros((4, 4), dtype=int),
                    "random_range": 1,
                    "n": 3,
                },
                "expected_output_shape": (3, 4, 4),
                "expected_output": np.zeros((3, 4, 4), dtype=int),
                "error": None,
            },
            {
                "name": "single_sensor",
                "input": {
                    "sensor_position_mask": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                    "random_range": 0,
                    "n": 2,
                },
                "error": ValueError,
            },
            {
                "name": "invalid_random_range",
                "input": {
                    "sensor_position_mask": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                    "random_range": -1,
                    "n": 2,
                },
                "expected_output": None,
                "error": ValueError,
            },
        ]

    def test_generate_vibrated_masks(self):
        for case in self.test_cases:
            with self.subTest(case["name"]):
                if case["error"]:
                    with self.assertRaises(case["error"]):
                        generate_vibrated_masks(**case["input"])
                else:
                    result = generate_vibrated_masks(**case["input"])
                    print(result)
                    self.assertEqual(result.shape, case["expected_output_shape"])
                    if "expected_output" in case:
                        np.testing.assert_array_equal(result, case["expected_output"])


if __name__ == "__main__":
    unittest.main()
