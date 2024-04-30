from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import Any


class BaseModel(metaclass=ABCMeta):
    def __init__(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = device

    @abstractmethod
    def predict(self, image: npt.NDArray[np.uint8]) -> dict[str, Any]:
        raise NotImplementedError
