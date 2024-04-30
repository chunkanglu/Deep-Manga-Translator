from abc import ABCMeta, abstractmethod
from PIL import Image, ImageDraw
import numpy as np
import numpy.typing as npt
from typing import Any, Callable, Union

from src.segmentation.basemodel import BaseModel
from src.utils import (
    DeviceEnum,
    get_crop,
    get_text,
    get_text_box,
    get_tr_text,
    draw_text,
    expand_text_box,
    process_ocr_text,
    ocr_bbox_sort,
)


class BaseProcessor(metaclass=ABCMeta):
    def __init__(
        self,
        seg_model: BaseModel,
        inpaint_model,
        translator,
        ocr_model,
        device: DeviceEnum,
    ) -> None:
        self.seg_model = seg_model
        self.inpaint_model = inpaint_model
        self.translator = translator
        self.ocr_model = ocr_model
        self.device = device.value

        self.last_image = None
        self.prediction = None

    def cache_prediction(self, image: npt.NDArray[np.uint8]) -> dict[str, Any]:
        if (self.last_image is None) or (not np.array_equal(self.last_image, image)):
            self.last_image = image
            self.prediction = self.seg_model.predict(image)
        assert self.prediction is not None
        return self.prediction

    @abstractmethod
    def clean_text(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        raise NotImplementedError

    def add_translated_text_process(
        self,
        image: npt.NDArray[np.uint8],
        clean_image: npt.NDArray[np.uint8],
        data: list[
            tuple[Union[npt.NDArray[np.bool_], None], tuple[int, int, int, int]]
        ],
        font_path: str,
        draw_text_logic: Callable
    ) -> Image.Image:
        output_image = Image.fromarray(clean_image)
        draw = ImageDraw.Draw(output_image)

        data = sorted(data, key=ocr_bbox_sort)
        masks_data = [m for m, _ in data]
        bbox_data = [b for _, b in data]

        # # Translate all at once
        # TODO: Translation with context
        # TODO: See if there is a better separator invariant to translation changes
        # SEP = "¶"
        # to_translate = ""
        # for bbox in bbox_data:
        #     crop = get_crop(image, bbox)
        #     og_text = process_ocr_text(get_text(crop, self.ocr_model))
        #     to_translate += og_text + SEP

        # # Split back
        # tr_text = get_tr_text(to_translate, self.translator)
        # tr_text = tr_text.split(SEP)

        # for mask, bbox, text in zip(masks_data, bbox_data, tr_text):
        #     draw_text_logic(draw, mask, bbox, text)

        for mask, bbox in data:
            crop = get_crop(image, bbox)
            og_text = process_ocr_text(get_text(crop, self.ocr_model))
            text = get_tr_text(og_text, self.translator)

            draw_text_logic(draw, mask, bbox, text)

        return output_image

    @abstractmethod
    def add_translated_text(
        self,
        image: npt.NDArray[np.uint8],
        clean_image: npt.NDArray[np.uint8],
        font_path: str,
    ) -> Image.Image:
        raise NotImplementedError
