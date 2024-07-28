import unittest
from unittest.mock import patch, mock_open
import yaml

from aidtep.utils.config import AidtepConfig


def check_file_exist(path):
    return path == "config.yaml"


class TestAidtepConfig(unittest.TestCase):

    def setUp(self):
        self.test_cases = {
            "test_init_with_config_path": [
                {
                    "name": "valid_config_path",
                    "input": {"config_path": "config.yaml"},
                    "mock_data": yaml.dump({"key1": {"key2": "value"}}),
                    "expected_output": "AidtepConfig({'key1': AidtepConfig({'key2': 'value'})})",
                    "error": None
                },
                {
                    "name": "invalid_config_path",
                    "input": {"config_path": "invalid.yaml"},
                    "mock_data": None,
                    "expected_output": None,
                    "error": FileNotFoundError
                }
            ],
            "test_init_with_config_dict": [
                {
                    "name": "valid_config_dict",
                    "input": {"config_dict": {"key1": {"key2": "value"}}},
                    "expected_output": "AidtepConfig({'key1': AidtepConfig({'key2': 'value'})})",
                    "error": None
                }
            ],
            "test_get_method": [
                {
                    "name": "get_existing_key",
                    "input": {
                        "config_dict": {"key1": {"key2": "value"}},
                        "key": "key1.key2",
                        "default": None
                    },
                    "expected_output": "value",
                    "error": None
                },
                {
                    "name": "get_non_existing_key",
                    "input": {
                        "config_dict": {"key1": {"key2": "value"}},
                        "key": "key1.key3",
                        "default": "default_value"
                    },
                    "expected_output": "default_value",
                    "error": None
                }
            ],
            "test_keys_method": [
                {
                    "name": "keys_method",
                    "input": {"config_dict": {"key1": {"key2": "value"}, "key3": "value3"}},
                    "expected_output": ["key1", "key3"],
                    "error": None
                }
            ]
        }

    @patch("builtins.open", new_callable=mock_open, read_data=yaml.dump({"key1": {"key2": "value"}}))
    @patch("aidtep.utils.config.check_file_exist", side_effect=lambda path: path == "config.yaml")
    def test_init_with_config_path(self, mock_check_file_exist, mock_file):
        for case in self.test_cases["test_init_with_config_path"]:
            with self.subTest(case["name"]):
                if case["error"]:
                    with self.assertRaises(case["error"]):
                        AidtepConfig(config_path=case["input"]["config_path"])
                else:
                    config = AidtepConfig(config_path=case["input"]["config_path"])
                    self.assertEqual(repr(config), case["expected_output"])

    def test_init_with_config_dict(self):
        for case in self.test_cases["test_init_with_config_dict"]:
            with self.subTest(case["name"]):
                if case["error"]:
                    with self.assertRaises(case["error"]):
                        AidtepConfig(config_dict=case["input"]["config_dict"])
                else:
                    config = AidtepConfig(config_dict=case["input"]["config_dict"])
                    self.assertEqual(repr(config), case["expected_output"])

    def test_get_method(self):
        for case in self.test_cases["test_get_method"]:
            with self.subTest(case["name"]):
                config = AidtepConfig(config_dict=case["input"]["config_dict"])
                result = config.get(case["input"]["key"], case["input"]["default"])
                self.assertEqual(result, case["expected_output"])

    def test_keys_method(self):
        for case in self.test_cases["test_keys_method"]:
            with self.subTest(case["name"]):
                config = AidtepConfig(config_dict=case["input"]["config_dict"])
                result = config.keys()
                self.assertEqual(result, case["expected_output"])


if __name__ == "__main__":
    unittest.main()
