import largestinteriorrectangle as lir
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
import re
import functools
import torchvision.transforms as T

from src.processor.baseprocessor import BaseProcessor
# from src.segmentation.detectron_bubble_seg import Detectron2BubbleSegmentationModel
# from src.segmentation.pytorch_bubble_seg import PytorchBubbleSegmentationModel
from src.utils import get_crop, get_text, get_tr_text, draw_text

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
        
        def get_largest_text_box(mask: npt.NDArray[np.bool_]
                                    ) -> npt.NDArray[np.uint32]:
                return lir.lir(mask).astype(np.uint32)

        output_image = Image.fromarray(clean_image.copy())
        draw = ImageDraw.Draw(output_image)

        preds = self.seg_model.predict(image)

        for mask, bbox in zip(preds["masks"], preds["bboxs"]):
            draw = ImageDraw.Draw(output_image)

            crop = get_crop(image, bbox)

            og_text = get_text(crop, self.ocr_model)
            tr_text = get_tr_text(og_text, self.translator)

            x, y, w, h = get_largest_text_box(mask)

            draw_text((x, y, x+w, y+h), tr_text, draw, font_path)

        return output_image