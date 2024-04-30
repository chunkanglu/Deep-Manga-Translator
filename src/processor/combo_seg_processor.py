import cv2
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw
import torchvision.transforms as T

from src.processor.baseprocessor import BaseProcessor
from src.segmentation.pytorch_bubble_seg import PytorchBubbleSegmentationModel
from src.segmentation.text_seg import TextSegmentationModel
from src.utils import DeviceEnum


class ComboSegProcessor(BaseProcessor):
    def __init__(
        self,
        bubble_seg_model: PytorchBubbleSegmentationModel,
        text_seg_model: TextSegmentationModel,
        inpaint_model,
        translator,
        ocr_model,
        device: DeviceEnum,
    ) -> None:
        super().__init__(bubble_seg_model, inpaint_model, translator, ocr_model, device)

        self.text_seg_model = text_seg_model
        self.bubble_prediction = None
        self.text_prediction = None

    def cache_prediction(
        self, image: npt.NDArray[np.uint8]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if (self.last_image is None) or (not np.array_equal(self.last_image, image)):
            self.last_image = image
            bubble_pred = self.seg_model.predict(image)
            text_pred = self.text_seg_model.predict(image)

            bubble_masks = []
            bubble_bboxs = []
            bubble_text_bboxs = []
            text_mask = text_pred["mask"].copy()
            text_bboxs = text_pred["bboxs"].copy()

            # Only keep bubbles with text inside, remove text mask area for those
            for mask, bbox in zip(bubble_pred["masks"], bubble_pred["bboxs"]):
                i = 0
                while i < len(text_bboxs):
                    x1, y1, x2, y2 = text_bboxs[i]
                    if np.any(
                        np.logical_and(mask[y1:y2, x1:x2], text_mask[y1:y2, x1:x2])
                    ):
                        bubble_masks.append(mask)
                        bubble_bboxs.append(bbox)
                        bubble_text_bboxs.append((x1, y1, x2, y2))

                        dilated_mask = cv2.morphologyEx(
                            mask.astype(np.uint8),
                            cv2.MORPH_DILATE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                            iterations=10,
                        )
                        dilated_mask = (
                            cv2.threshold(dilated_mask, 0.5, 1, cv2.THRESH_BINARY)[
                                1
                            ].astype(np.uint8)
                            > 0.5
                        )
                        text_mask[dilated_mask] = False

                        text_bboxs.pop(i)
                        break
                    i += 1

            # Save predictions
            self.bubble_prediction = {
                "masks": bubble_masks,
                "bboxs": bubble_bboxs,
                "text_bboxs": bubble_text_bboxs,
            }
            self.text_prediction = {
                "og_mask": text_pred["mask"],
                "mask": text_mask,
                "bboxs": text_bboxs,
            }

        assert (self.bubble_prediction is not None) and (
            self.text_prediction is not None
        )
        return (self.bubble_prediction, self.text_prediction)

    def clean_text(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        bubble_preds, text_preds = self.cache_prediction(image)
        # image[text_preds["mask"], :] = (255, 255, 255)
        # for mask in bubble_preds["masks"]:
        #     image[mask, :] = (255, 255, 255)
        if self.inpaint_model is None:
            image = image.copy()
            image[text_preds["og_mask"], :] = (255, 255, 255)
            return image

        image_t = T.ToTensor()(image).to(self.device)
        mask_t = T.ToTensor()(text_preds["og_mask"]).to(self.device)
        return self.inpaint_model.predict(image_t, mask_t)

    def add_translated_text(
        self,
        image: npt.NDArray[np.uint8],
        clean_image: npt.NDArray[np.uint8],
        font_path: str,
    ) -> Image.Image:
        bubble_preds, text_preds = self.cache_prediction(image)

        text_data = list(zip([None] * len(text_preds["bboxs"]), text_preds["bboxs"]))
        bubble_data = list(zip(bubble_preds["masks"], bubble_preds["text_bboxs"]))
        all_data = text_data + bubble_data

        return self.add_translated_text_process(image, clean_image, all_data, font_path)
