import unittest

import numpy as np

from aidtep.data_process.component.noise import add_noise


class TestAddNoise(unittest.TestCase):

    def setUp(self):
        self.test_cases = [
            {
                "name": "normal_case_1d",
                "input": {"data": np.array([1.0, 2.0, 3.0]), "noise_ratio": 0.1},
                "expected_output_shape": (3,),
                "error": None,
            },
            {
                "name": "normal_case_2d",
                "input": {
                    "data": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                    "noise_ratio": 0.2,
                },
                "expected_output_shape": (2, 3),
                "error": None,
            },
            {
                "name": "normal_case_3d",
                "input": {
                    "data": np.array(
                        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
                    ),
                    "noise_ratio": 0.3,
                },
                "expected_output_shape": (2, 2, 2),
                "error": None,
            },
            {
                "name": "zero_noise_ratio",
                "input": {"data": np.array([1.0, 2.0, 3.0]), "noise_ratio": 0.0},
                "expected_output_shape": (3,),
                "expected_output": np.array([1.0, 2.0, 3.0]),
                "error": None,
            },
            {
                "name": "negative_noise_ratio",
                "input": {"data": np.array([1.0, 2.0, 3.0]), "noise_ratio": -0.1},
                "expected_output_shape": (3,),
                "error": ValueError,
            },
            {
                "name": "invalid_data_type",
                "input": {"data": [1.0, 2.0, 3.0], "noise_ratio": 0.1},
                "expected_output": None,
                "error": TypeError,
            },
        ]

    def test_add_noise(self):
        for case in self.test_cases:
            with self.subTest(case["name"]):
                if case["error"]:
                    with self.assertRaises(case["error"]):
                        add_noise(case["input"]["data"], case["input"]["noise_ratio"])
                else:
                    result = add_noise(
                        case["input"]["data"], case["input"]["noise_ratio"]
                    )
                    self.assertEqual(result.shape, case["expected_output_shape"])
                    if "expected_output" in case:
                        np.testing.assert_array_almost_equal(
                            result, case["expected_output"], decimal=5
                        )


if __name__ == "__main__":
    unittest.main()
