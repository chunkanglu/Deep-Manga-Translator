import numpy as np
import numpy.typing as npt
from typing import Any

class BaseModel():
    def __init__(self,
                 model_path: str,
                 device: str) -> None:
        self.model_path = model_path
        self.device = device

    def predict(self,
                image: npt.NDArray[np.uint8]
                ) -> dict[str, Any]:
        raise NotImplementedError
