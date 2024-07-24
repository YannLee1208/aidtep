from aidtep.utils.config import init_config, AidtepConfig

import unittest
from unittest.mock import patch, mock_open
import yaml


class MockConfig:
    def __init__(self, data):
        self.data = data

    def asdict(self):
        return self.data


class TestAidtepConfig(unittest.TestCase):

    def setUp(self):
        self.test_load_config_cases = [
            {
                "name": "load_config_success",
                "input": "config.yaml",
                "expected_output": {"database": {"host": "localhost", "port": 5432},
                                    "api": {"endpoint": "/api/v1"}},
                "mock_data": yaml.dump(
                    {"database": {"host": "localhost", "port": 5432}, "api": {"endpoint": "/api/v1"}}),
                "error": None
            }
        ]

        self.test_singleton_behavior_cases = [
            {
                "name": "singleton_test",
                "input": "config.yaml",
                "expected_output": {"database": {"host": "localhost", "port": 5432},
                                    "api": {"endpoint": "/api/v1"}},
                "mock_data": yaml.dump(
                    {"database": {"host": "localhost", "port": 5432}, "api": {"endpoint": "/api/v1"}}),
                "error": None
            }
        ]

        self.test_get_cases = [
            {
                "name": "get_existing_key",
                "input": ("database.host", "config.yaml"),
                "expected_output": "localhost",
                "mock_data": yaml.dump(
                    {"database": {"host": "localhost", "port": 5432}, "api": {"endpoint": "/api/v1"}}),
                "error": None
            },
            {
                "name": "get_non_existing_key",
                "input": ("non.existent.key", "config.yaml"),
                "expected_output": None,
                "mock_data": yaml.dump(
                    {"database": {"host": "localhost", "port": 5432}, "api": {"endpoint": "/api/v1"}}),
                "error": None
            },
            {
                "name": "get_non_existing_key_with_default",
                "input": ("non.existent.key", "config.yaml"),
                "expected_output": "default",
                "mock_data": yaml.dump(
                    {"database": {"host": "localhost", "port": 5432}, "api": {"endpoint": "/api/v1"}}),
                "error": None
            }
        ]

        self.test_getitem_cases = [
            {
                "name": "getitem_existing_key",
                "input": ("database.host", "config.yaml"),
                "expected_output": "localhost",
                "mock_data": yaml.dump(
                    {"database": {"host": "localhost", "port": 5432}, "api": {"endpoint": "/api/v1"}}),
                "error": None
            }
        ]

        self.test_init_config_cases = [
            {
                "name": "init_config_success",
                "input": "config.yaml",
                "expected_output": {"database": {"host": "localhost", "port": 5432},
                                    "api": {"endpoint": "/api/v1"}},
                "mock_data": yaml.dump(
                    {"database": {"host": "localhost", "port": 5432}, "api": {"endpoint": "/api/v1"}}),
                "error": None,
                "mock_exists": True
            },
            {
                "name": "init_config_file_not_found",
                "input": "config.yaml",
                "expected_output": None,
                "mock_data": None,
                "error": FileNotFoundError,
                "mock_exists": False
            },
        ]
        AidtepConfig._instance = None

    def test_load_config(self):
        for case in self.test_load_config_cases:
            with self.subTest(case["name"]):
                with patch("builtins.open", new_callable=mock_open, read_data=case["mock_data"]):
                    config = AidtepConfig(case["input"])
                    self.assertEqual(config.config, case["expected_output"])

    def test_singleton_behavior(self):
        for case in self.test_singleton_behavior_cases:
            with self.subTest(case["name"]):
                with patch("builtins.open", new_callable=mock_open, read_data=case["mock_data"]):
                    config1 = AidtepConfig(case["input"])
                    config2 = AidtepConfig(case["input"])
                    self.assertIs(config1, config2)
                    self.assertEqual(config1.config, case["expected_output"])

    def test_get(self):
        for case in self.test_get_cases:
            with self.subTest(case["name"]):
                with patch("builtins.open", new_callable=mock_open, read_data=case["mock_data"]):
                    config = AidtepConfig(case["input"][1])
                    if case["name"] == "get_non_existing_key_with_default":
                        self.assertEqual(config.get(case["input"][0], "default"), case["expected_output"])
                    else:
                        self.assertEqual(config.get(case["input"][0]), case["expected_output"])

    def test_getitem(self):
        for case in self.test_getitem_cases:
            with self.subTest(case["name"]):
                with patch("builtins.open", new_callable=mock_open, read_data=case["mock_data"]):
                    config = AidtepConfig(case["input"][1])
                    self.assertEqual(config[case["input"][0]], case["expected_output"])

    def test_init_config(self):
        for case in self.test_init_config_cases:
            with self.subTest(case["name"]):
                with patch("os.path.exists", return_value=case["mock_exists"]):
                    if case["error"]:
                        with self.assertRaises(case["error"]):
                            init_config(case["input"])
                    else:
                        with patch("builtins.open", new_callable=mock_open, read_data=case["mock_data"]):
                            config = init_config(case["input"])
                            self.assertEqual(config.config, case["expected_output"])


if __name__ == "__main__":
    unittest.main()
