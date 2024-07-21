import unittest
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from aidtep.ml.models.sklearn_model import SklearnModel


class TestSklearnModel(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y_train = np.array([0, 0, 1, 1])
        self.X_test = np.array([[1, 2], [2, 3]])
        self.y_test = np.array([0, 0])

        self.svc_model = SklearnModel(SVC())
        self.knn_model = SklearnModel(KNeighborsRegressor(n_neighbors=2, weights='distance',
                                                    metric='minkowski'))

    def test_svc_train(self):
        try:
            self.svc_model.train(self.X_train, self.y_train)
        except Exception as e:
            self.fail(f"SVC model training failed with exception {e}")

    def test_svc_predict(self):
        self.svc_model.train(self.X_train, self.y_train)
        predictions = self.svc_model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

    def test_svc_evaluate(self):
        self.svc_model.train(self.X_train, self.y_train)
        score = self.svc_model.evaluate(self.X_test, self.y_test)
        self.assertIsInstance(score, float)

    def test_knn_train(self):
        try:
            self.knn_model.train(self.X_train, self.y_train)
        except Exception as e:
            self.fail(f"KNN model training failed with exception {e}")

    def test_knn_predict(self):
        self.knn_model.train(self.X_train, self.y_train)
        predictions = self.knn_model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

    def test_knn_evaluate(self):
        self.knn_model.train(self.X_train, self.y_train)
        score = self.knn_model.evaluate(self.X_test, self.y_test)
        self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()