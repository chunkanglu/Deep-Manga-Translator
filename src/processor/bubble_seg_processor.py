import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
import functools
import torchvision.transforms as T

from src.processor.baseprocessor import BaseProcessor


class BubbleSegProcessor(BaseProcessor):
    def __init__(self,
                 seg_model,
                 inpaint_model,
                 translator,
                 ocr_model,
                 device) -> None:
        super().__init__(seg_model, inpaint_model, translator, ocr_model, device)

    # TODO: untested
    def clean_text(self,
                   image: npt.NDArray[np.uint8]
                   ) -> npt.NDArray[np.uint8]:
        preds = self.seg_model.predict(image)

        if self.inpaint_model is None:
            image = image.copy()
            for mask in preds["masks"]:
                image[mask, :] = [255, 255, 255]
            return image

        combined_mask = functools.reduce(np.logical_or, preds["masks"])
        image_t = T.ToTensor()(image).to(self.device)
        mask_t = T.ToTensor()(combined_mask).to(self.device)
        return self.inpaint_model.predict(image_t, mask_t)

    def add_translated_text(self,
                            image: npt.NDArray[np.uint8],
                            clean_image: npt.NDArray[np.uint8],
                            font_path: str
                            ) -> Image.Image:

        preds = self.cache_prediction(image)
        data = list(zip(preds["masks"], preds["bboxs"]))

        return self.add_translated_text_process(image,
                                                clean_image,
                                                data,
                                                font_path)
