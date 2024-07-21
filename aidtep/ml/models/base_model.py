from abc import ABC, abstractmethod
from typing import Any
from loguru import logger
from sklearn.base import BaseEstimator
import joblib
import numpy as np
        
import torch
import torch.nn as nn
import torch.optim as optim
import pickle


class BaseModel(ABC):
    def __init__(self):
        self.logger = logger

    @abstractmethod
    def train(self, X: Any, y: Any, **kwargs) -> None:
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        pass

    @abstractmethod
    def evaluate(self, X: Any, y: Any, **kwargs) -> float:
        pass

    @abstractmethod
    def save_model(self, filepath: str) -> None:
        pass

    @abstractmethod
    def load_model(self, filepath: str) -> None:
        pass
    




