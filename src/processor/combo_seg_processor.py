import cv2
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw
import torchvision.transforms as T

from src.processor.baseprocessor import BaseProcessor
from src.segmentation.pytorch_bubble_seg import PytorchBubbleSegmentationModel
from src.segmentation.text_seg import TextSegmentationModel


class ComboSegProcessor(BaseProcessor):
    def __init__(self,
                 bubble_seg_model: PytorchBubbleSegmentationModel,
                 text_seg_model: TextSegmentationModel,
                 inpaint_model,
                 translator,
                 ocr_model,
                 device) -> None:
        super().__init__(bubble_seg_model, inpaint_model, translator, ocr_model, device)

        self.text_seg_model = text_seg_model
        self.bubble_prediction = None
        self.text_prediction = None

    def cache_prediction(self,
                         image: npt.NDArray[np.uint8]
                         ) -> tuple[dict[str, Any],
                                    dict[str, Any]]:
        if (self.last_image is None) or (not np.array_equal(self.last_image, image)):
            self.last_image = image
            bubble_pred = self.seg_model.predict(image)
            text_pred = self.text_seg_model.predict(image)

            bubble_masks = []
            bubble_bboxs = []
            text_mask = text_pred["mask"].copy()
            text_bboxes = []

            # Only keep bubbles with text inside, remove text mask area for those
            for mask, bbox in zip(bubble_pred["masks"], bubble_pred["bboxs"]):
                intersection = np.logical_and(mask, text_mask)

                if np.any(intersection):
                    bubble_masks.append(mask)
                    bubble_bboxs.append(bbox)

                    dialated_mask = cv2.morphologyEx(mask.astype(np.uint8),
                                                     cv2.MORPH_DILATE,
                                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                               (5, 5)),
                                                     iterations=10)
                    dialated_mask = cv2.threshold(dialated_mask,
                                                  0.5,
                                                  1,
                                                  cv2.THRESH_BINARY)[1].astype(np.uint8) > 0.5
                    text_mask[dialated_mask] = False

            # Regenerate new text bboxs
            dialated = cv2.morphologyEx(text_mask.astype(np.uint8),
                                        cv2.MORPH_DILATE,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                  (3, 3)),
                                        iterations=5)

            thresh = cv2.threshold(dialated, 0.5, 1, cv2.THRESH_BINARY)[
                1].astype(np.uint8)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh,
                                                                       connectivity=8)
            for label in range(1, num_labels):
                x1 = stats[label, cv2.CC_STAT_LEFT]
                y1 = stats[label, cv2.CC_STAT_TOP]
                x2 = x1 + stats[label, cv2.CC_STAT_WIDTH]
                y2 = y1 + stats[label, cv2.CC_STAT_HEIGHT]

                text_bboxes.append((x1, y1, x2, y2))

            # Save predictions
            self.bubble_prediction = {
                "masks": bubble_masks,
                "bboxs": bubble_bboxs
            }
            self.text_prediction = {
                "og_mask": text_pred["mask"],
                "mask": text_mask,
                "bboxs": text_bboxes
            }

        assert ((self.bubble_prediction is not None) and
                (self.text_prediction is not None))
        return (self.bubble_prediction,
                self.text_prediction)

    def clean_text(self,
                   image: npt.NDArray[np.uint8]
                   ) -> npt.NDArray[np.uint8]:
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

    def add_translated_text(self,
                            image: npt.NDArray[np.uint8],
                            clean_image: npt.NDArray[np.uint8],
                            font_path: str
                            ) -> Image.Image:

        bubble_preds, text_preds = self.cache_prediction(image)

        # Sort data in reading order
        text_data = list(
            zip([None]*len(text_preds["bboxs"]), text_preds["bboxs"]))
        bubble_data = list(zip(bubble_preds["masks"], bubble_preds["bboxs"]))
        all_data = text_data + bubble_data

        return self.add_translated_text_process(image,
                                                clean_image,
                                                all_data,
                                                font_path)
