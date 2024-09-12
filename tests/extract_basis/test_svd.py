import unittest
import torch

from aidtep.extract_basis.svd import SVD


class TestSVD(unittest.TestCase):

    def setUp(self):
        self.base_number = 5
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.svd_extractor = SVD(self.base_number, self.device)

        # Create a mock full matrix
        self.full_matrix = torch.randn(100, 50)

    def test_extract_basis_and_importance(self):
        # Test extract method and check basis and basis importance
        self.svd_extractor.extract(self.full_matrix)

        basis = self.svd_extractor.get_basis()
        basis_importance = self.svd_extractor.get_basis_importance()

        # Check if basis and basis_importance are not None
        self.assertIsNotNone(basis)
        self.assertIsNotNone(basis_importance)

        # Check if the shapes of basis and basis_importance are correct
        self.assertEqual(basis.shape, (self.base_number, self.full_matrix.shape[1]))
        self.assertEqual(basis_importance.shape[0], self.base_number)

        # Check if the sum of basis_importance is close to 1
        self.assertAlmostEqual(torch.sum(basis_importance).item(), 1.0, places=5)

    def test_get_basis_before_extract(self):
        # Test get_basis method before extraction
        with self.assertRaises(ValueError):
            self.svd_extractor.get_basis()

    def test_get_basis_importance_before_extract(self):
        # Test get_basis_importance method before extraction
        with self.assertRaises(ValueError):
            self.svd_extractor.get_basis_importance()

    def test_name_method(self):
        # Test name method
        self.assertEqual(SVD.name(), "SVD")


if __name__ == "__main__":
    unittest.main()