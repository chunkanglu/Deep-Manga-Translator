from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import Any

from src.utils import DeviceEnum


class BaseModel(metaclass=ABCMeta):
    def __init__(self, model_path: str, device: DeviceEnum) -> None:
        self.model_path = model_path
        self.device = device.value

    @abstractmethod
    def predict(self, image: npt.NDArray[np.uint8]) -> dict[str, Any]:
        raise NotImplementedError
