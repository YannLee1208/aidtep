import os
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor

from aidtep.ml.models.base_models.sklearn_model import SklearnModel

import unittest
import torch
from torch.optim import SGD
from torch.nn import MSELoss
from aidtep.ml.models.base_models.torch_model import PyTorchModel


class TestSklearnModel(unittest.TestCase):
    def setUp(self):
        self.X_train = np.random.rand(100, 30)
        self.y_train = np.random.randint(0, 2, size=100)
        self.X_test = np.random.rand(20, 30)
        self.y_test = np.random.randint(0, 2, size=20)

        self.svc_model = SklearnModel(SVC())
        self.knn_model = SklearnModel(KNeighborsRegressor(n_neighbors=2, weights='distance',
                                                          metric='minkowski'))

        self.svc_filepath = "svc_model.pkl"
        self.knn_filepath = "knn_model.pkl"

    def tearDown(self):
        if os.path.exists(self.svc_filepath):
            os.remove(self.svc_filepath)
        if os.path.exists(self.knn_filepath):
            os.remove(self.knn_filepath)

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

    def test_svc_save_load(self):
        self.svc_model.train(self.X_train, self.y_train)
        original_predictions = self.svc_model.predict(self.X_test)
        self.svc_model.save_model(self.svc_filepath)
        self.svc_model.load_model(self.svc_filepath)
        loaded_predictions = self.svc_model.predict(self.X_test)
        np.testing.assert_array_equal(original_predictions, loaded_predictions,
                                      "Predictions after loading model do not match original predictions.")

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

    def test_knn_save_load(self):
        self.knn_model.train(self.X_train, self.y_train)
        original_predictions = self.knn_model.predict(self.X_test)
        self.knn_model.save_model(self.knn_filepath)
        self.knn_model.load_model(self.knn_filepath)
        loaded_predictions = self.knn_model.predict(self.X_test)
        np.testing.assert_array_equal(original_predictions, loaded_predictions,
                                      "Predictions after loading model do not match original predictions.")


class SimpleLinearRegression(torch.nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.linear = torch.nn.Linear(30 * 30, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.linear(x)


class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(30 * 30, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestPyTorchModel(unittest.TestCase):
    def setUp(self):
        self.X_train = torch.rand(100, 30, 30)
        self.y_train = torch.rand(100, 10)
        self.X_test = torch.rand(20, 30, 30)
        self.y_test = torch.rand(20, 10)

        self.linear_model = PyTorchModel(SimpleLinearRegression(), MSELoss(),
                                         SGD(SimpleLinearRegression().parameters(), lr=0.01))
        self.nn_model = PyTorchModel(SimpleNN(), MSELoss(), SGD(SimpleNN().parameters(), lr=0.01))
        self.linear_filepath = "linear_model.pth"
        self.nn_filepath = "nn_model.pth"

    def tearDown(self):
        if os.path.exists(self.linear_filepath):
            os.remove(self.linear_filepath)
        if os.path.exists(self.nn_filepath):
            os.remove(self.nn_filepath)

    def test_linear_model_train(self):
        try:
            self.linear_model.train(self.X_train, self.y_train, epochs=10, batch_size=2)
        except Exception as e:
            self.fail(f"Linear model training failed with exception {e}")

    def test_linear_model_predict(self):
        self.linear_model.train(self.X_train, self.y_train, epochs=10, batch_size=2)
        predictions = self.linear_model.predict(self.X_test)
        self.assertEqual(predictions.shape, self.y_test.shape)

    def test_linear_model_evaluate(self):
        self.linear_model.train(self.X_train, self.y_train, epochs=10, batch_size=2)
        loss = self.linear_model.evaluate(self.X_test, self.y_test)
        self.assertIsInstance(loss, float)

    def test_linear_model_save_load(self):
        self.linear_model.train(self.X_train, self.y_train, epochs=10, batch_size=2)
        original_predictions = self.linear_model.predict(self.X_test).detach().cpu().numpy()
        self.linear_model.save_model(self.linear_filepath)
        self.linear_model.load_model(self.linear_filepath)
        loaded_predictions = self.linear_model.predict(self.X_test).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions, decimal=5,
                                             err_msg="Predictions after loading model do not match original predictions.")

    def test_nn_model_train(self):
        try:
            self.nn_model.train(self.X_train, self.y_train, epochs=10, batch_size=2)
        except Exception as e:
            self.fail(f"NN model training failed with exception {e}")

    def test_nn_model_predict(self):
        self.nn_model.train(self.X_train, self.y_train, epochs=10, batch_size=2)
        predictions = self.nn_model.predict(self.X_test)
        self.assertEqual(predictions.shape, self.y_test.shape)

    def test_nn_model_evaluate(self):
        self.nn_model.train(self.X_train, self.y_train, epochs=10, batch_size=2)
        loss = self.nn_model.evaluate(self.X_test, self.y_test)
        self.assertIsInstance(loss, float)

    def test_nn_model_save_load(self):
        self.nn_model.train(self.X_train, self.y_train, epochs=10, batch_size=2)
        original_predictions = self.nn_model.predict(self.X_test).detach().cpu().numpy()
        self.nn_model.save_model(self.nn_filepath)
        self.nn_model.load_model(self.nn_filepath)
        loaded_predictions = self.nn_model.predict(self.X_test).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions, decimal=5,
                                             err_msg="Predictions after loading model do not match original predictions.")


if __name__ == "__main__":
    unittest.main()
