from typing import Any, Union

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

from src.processor.baseprocessor import BaseProcessor
from src.segmentation.text_seg import (
    TextSegmentationModel,
    ThresholdTextSegmentationModel,
)
from src.utils import DeviceEnum


class TextSegProcessor(BaseProcessor):
    def __init__(
        self,
        seg_model: Union[TextSegmentationModel, ThresholdTextSegmentationModel],
        inpaint_model,
        translator,
        ocr_model,
        device: DeviceEnum,
    ) -> None:
        super().__init__(seg_model, inpaint_model, translator, ocr_model, device)

    def clean_text(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        mask = self.cache_prediction(image)["mask"]

        if self.inpaint_model is None:
            image = image.copy()
            image[mask, :] = [255, 255, 255]
            return image

        image_t = T.ToTensor()(image).to(self.device)
        mask_t = T.ToTensor()(mask).to(self.device)
        return self.inpaint_model.predict(image_t, mask_t)

    def add_translated_text(
        self,
        image: npt.NDArray[np.uint8],
        clean_image: npt.NDArray[np.uint8],
        font_path: str,
    ) -> Image.Image:
        bboxs = self.cache_prediction(image)["bboxs"]

        data = list(zip([None] * len(bboxs), bboxs))

        return self.add_translated_text_process(image, clean_image, data, font_path)
