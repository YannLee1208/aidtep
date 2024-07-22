from abc import ABC, abstractmethod
from typing import Any
from loguru import logger


class BaseModel(ABC):
    """
    Base class for all models. Now it is an abstract class.
    """
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
