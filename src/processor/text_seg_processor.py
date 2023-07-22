import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
import re
from typing import Any

from processor.baseprocessor import BaseProcessor
from segmentation.text_seg import TextSegmentationModel
from utils import get_crop, get_text, get_tr_text, draw_text

class TextSegProcessor(BaseProcessor):
    def __init__(self,
                 seg_model: TextSegmentationModel,
                 inpaint_model,
                 translator,
                 ocr_model) -> None:
        super().__init__(seg_model, inpaint_model, translator, ocr_model)

    def clean_text(self,
                   image: npt.NDArray[np.uint8]
                   ) -> npt.NDArray[np.uint8]:
        mask = self.cache_prediction(image)["mask"]
        image = image.copy()
        image[mask, :] = [255, 255, 255]
        return image
    
    def add_translated_text(self,
                            image: npt.NDArray[np.uint8],
                            clean_image: npt.NDArray[np.uint8],
                            font_path: str
                            ) -> Image.Image:
        bboxs = self.cache_prediction(image)["bboxs"]

        output_image = Image.fromarray(clean_image)
        draw = ImageDraw.Draw(output_image)

        for bbox in bboxs:
            draw = ImageDraw.Draw(output_image)

            crop = get_crop(image, bbox)

            og_text = get_text(crop, self.ocr_model)
            tr_text = get_tr_text(og_text, self.translator)

            draw_text(bbox, tr_text, draw, font_path)

        return output_image
