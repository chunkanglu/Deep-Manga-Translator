from abc import ABCMeta, abstractmethod
from PIL import Image
import numpy as np
import numpy.typing as npt
from typing import Any

from src.segmentation.basemodel import BaseModel

class BaseProcessor(metaclass=ABCMeta):
    def __init__(self,
                 seg_model: BaseModel,
                 inpaint_model,
                 translator,
                 ocr_model,
                 device) -> None:
        self.seg_model = seg_model
        self.inpaint_model = inpaint_model
        self.translator = translator
        self.ocr_model = ocr_model
        self.device = device

        self.last_image = None
        self.prediction = None

    def cache_prediction(self,
                         image: npt.NDArray[np.uint8]
                         ) -> dict[str, Any]:
        if (self.last_image is None) or (not np.array_equal(self.last_image, image)):
            self.last_image = image
            self.prediction = self.seg_model.predict(image)
        assert self.prediction is not None
        return self.prediction

    @abstractmethod
    def clean_text(self,
                   image: npt.NDArray[np.uint8]
                   ) -> npt.NDArray[np.uint8]:
        raise NotImplementedError
    
    @ abstractmethod
    def add_translated_text(self,
                            image: npt.NDArray[np.uint8],
                            clean_image: npt.NDArray[np.uint8],
                            font_path: str
                            ) -> Image.Image:
        raise NotImplementedError