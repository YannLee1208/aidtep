from aidtep.data_process.common import normalize_2d_array

import unittest
import numpy as np


class TestNormalize2DArray(unittest.TestCase):

    def setUp(self):
        self.test_cases = [
            {
                "name": "normal_case",
                "input": np.array([[1, 2], [3, 4]]),
                "expected_output": (np.array([[0.25, 0.33333333], [0.75, 0.66666667]]), np.array([4, 6])),
                "mock": None,
                "error": None
            },
            {
                "name": "non_np_array",
                "input": [[1, 2], [3, 4]],
                "expected_output": None,
                "mock": None,
                "error": ValueError
            },
            {
                "name": "non_2d_array",
                "input": np.array([1, 2, 3, 4]),
                "expected_output": None,
                "mock": None,
                "error": ValueError
            }
        ]

    def test_normalize_2d_array(self):
        for case in self.test_cases:
            with self.subTest(case["name"]):
                if case["error"]:
                    with self.assertRaises(case["error"]):
                        normalize_2d_array(case["input"])
                else:
                    result = normalize_2d_array(case["input"])
                    np.testing.assert_array_almost_equal(result[0], case["expected_output"][0])
                    np.testing.assert_array_almost_equal(result[1], case["expected_output"][1])


if __name__ == "__main__":
    unittest.main()