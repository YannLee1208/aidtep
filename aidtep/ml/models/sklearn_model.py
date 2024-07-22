import joblib
import numpy as np
from sklearn.base import BaseEstimator

from aidtep.ml.models.base_model import BaseModel


class SklearnModel(BaseModel):
    """
    Wrapper class for sklearn models.
    """
    def __init__(self, model: BaseEstimator):
        super().__init__()
        self.model = model

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.logger.info("Training sklearn model...")
        self.model.fit(X, y, **kwargs)
        self.logger.info("Training completed.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.logger.info("Predicting with sklearn model...")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> float:
        self.logger.info("Evaluating sklearn model...")
        return self.model.score(X, y, **kwargs)

    def save_model(self, filepath: str) -> None:
        self.logger.info(f"Saving sklearn model to {filepath}...")
        joblib.dump(self.model, filepath)

    def load_model(self, filepath: str) -> None:
        self.logger.info(f"Loading sklearn model from {filepath}...")
        self.model = joblib.load(filepath)
        