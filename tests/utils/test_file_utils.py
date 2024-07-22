import unittest

from aidtep.utils.file_utils import check_file_exist


class TestFileUtils(unittest.TestCase):
    def setUp(self):
        self.file_exist_test_cases = [
            {
                "name": "file not exist",
                "input": "test.py",
                "expected": False
            },
            {
                "name": "file exist",
                "input": "__init__.py",
                "expected": True
            }
        ]

    def tearDown(self):
        pass

    def test_check_file_exist(self):
        for test_case in self.file_exist_test_cases:
            with self.subTest(name=test_case["name"]):
                self.assertEqual(check_file_exist(test_case["input"]), test_case["expected"])


if __name__ == '__main__':
    unittest.main()
