import unittest
from unittest.mock import patch, mock_open
import numpy as np
from pathlib import Path

from aidtep.utils.file import check_file_exist, make_parent_dir, save_ndarray


class TestFileUtilities(unittest.TestCase):

    def setUp(self):
        self.test_check_file_exist_cases = [
            {
                "name": "file_exists",
                "input": "existing_file.txt",
                "expected_output": True,
                "mock": {
                    "exists": True
                },
                "error": None
            },
            {
                "name": "file_does_not_exist",
                "input": "non_existing_file.txt",
                "expected_output": False,
                "mock": {
                    "exists": False
                },
                "error": None
            },
            {
                "name": "empty_file_path",
                "input": "",
                "expected_output": False,
                "mock": None,
                "error": None
            }
        ]

        self.test_make_parent_dir_cases = [
            {
                "name": "make_parent_dir",
                "input": "some_dir/some_file.txt",
                "expected_output": None,
                "mock": None,
                "error": None
            },
            {
                "name": "empty_file_path",
                "input": "",
                "expected_output": None,
                "mock": None,
                "error": ValueError
            }
        ]

        self.test_save_ndarray_cases = [
            {
                "name": "save_array",
                "input": {
                    "file_path": "array_dir/array_file.npy",
                    "data": np.array([1, 2, 3])
                },
                "expected_output": None,
                "mock": None,
                "error": None
            }
        ]

    @patch("pathlib.Path.exists")
    def test_check_file_exist(self, mock_exists):
        for case in self.test_check_file_exist_cases:
            with self.subTest(case["name"]):
                if case["mock"] and "exists" in case["mock"]:
                    mock_exists.return_value = case["mock"]["exists"]
                result = check_file_exist(case["input"])
                self.assertEqual(result, case["expected_output"])

    @patch("pathlib.Path.mkdir")
    def test_make_parent_dir(self, mock_mkdir):
        for case in self.test_make_parent_dir_cases:
            with self.subTest(case["name"]):
                if case["error"]:
                    with self.assertRaises(case["error"]):
                        make_parent_dir(case["input"])
                else:
                    make_parent_dir(case["input"])
                    if case["input"]:
                        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("pathlib.Path.mkdir")
    @patch("numpy.save")
    def test_save_ndarray(self, mock_save, mock_mkdir):
        for case in self.test_save_ndarray_cases:
            with self.subTest(case["name"]):
                save_ndarray(case["input"]["file_path"], case["input"]["data"])
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
                mock_save.assert_called_once_with(Path(case["input"]["file_path"]), case["input"]["data"])

if __name__ == "__main__":
    unittest.main()
