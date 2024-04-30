import albumentations as A
import cv2
import numpy as np
import numpy.typing as npt
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch
from typing import Any

from src.segmentation.basemodel import BaseModel
from src.utils import DeviceEnum


class TextSegmentationModel(BaseModel):
    def __init__(self, model_path: str, device: DeviceEnum) -> None:
        super().__init__(model_path, device)

        def to_tensor(x, **kwargs):
            return x.transpose(2, 0, 1).astype("float32")

        def to_tensor_mask(x, **kwargs):
            return np.expand_dims(x, axis=0).astype("float32")

        def get_preprocessing(preprocessing_fn):
            """Construct preprocessing transform

            Args:
                preprocessing_fn (callable): data normalization function
                    (can be specific for each pretrained neural network)
            Return:
                transform: albumentations.Compose

            """

            _transform = [
                A.Lambda(image=preprocessing_fn),
                A.Lambda(image=to_tensor, mask=to_tensor_mask),
            ]
            return A.Compose(_transform)

        preprocessing_fn = get_preprocessing_fn(
            encoder_name="resnet34", pretrained="imagenet"
        )

        self.preprocessing = get_preprocessing(preprocessing_fn)

        self.dilate = lambda x, iterations: cv2.morphologyEx(
            x,
            cv2.MORPH_DILATE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=iterations,
        )

        self.predictor = torch.load(self.model_path, map_location=self.device).to(self.device)

    def predict(self, image: npt.NDArray[np.uint8]) -> dict[str, Any]:
        og_shape = image.shape
        padded_height = og_shape[0] + 32 - (og_shape[0] % 32)
        padded_width = og_shape[1] + 32 - (og_shape[1] % 32)

        image = self.preprocessing(image=image)["image"]
        image = np.transpose(image, (1, 2, 0))
        image = A.PadIfNeeded(
            padded_height, padded_width, position=A.PadIfNeeded.PositionType.TOP_LEFT
        )(image=image)["image"]

        image_tensor = torch.from_numpy(image).to(self.device).unsqueeze(0)

        prediction = self.predictor.predict(image_tensor.permute(0, 3, 1, 2).float())

        pred_mask = prediction.squeeze().cpu().numpy()
        pred_mask = pred_mask[: og_shape[0], : og_shape[1]]
        og_mask = pred_mask
        pred_mask = self.dilate(pred_mask, 5)
        dilated = self.dilate(pred_mask, 7)

        pred_mask = pred_mask > 0.5

        thresh = cv2.threshold(dilated, 0.5, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            thresh, connectivity=8
        )
        bboxes = []
        for label in range(1, num_labels):
            x1 = stats[label, cv2.CC_STAT_LEFT]
            y1 = stats[label, cv2.CC_STAT_TOP]
            x2 = x1 + stats[label, cv2.CC_STAT_WIDTH]
            y2 = y1 + stats[label, cv2.CC_STAT_HEIGHT]

            bboxes.append((x1, y1, x2, y2))

        return {"og_mask": og_mask, "mask": pred_mask, "bboxs": bboxes}


class ThresholdTextSegmentationModel(TextSegmentationModel):
    def __init__(self, model_path: str, device: DeviceEnum) -> None:
        super().__init__(model_path, device)

    def predict(self, image: npt.NDArray[np.uint8]) -> dict[str, Any]:
        preds = super().predict(image)

        og_mask = preds["og_mask"].astype(np.uint8)
        mask = preds["mask"].astype(np.uint8)
        bboxs = preds["bboxs"]

        new_mask = np.zeros_like(mask)

        # Move mask down 3 pixels since prediction seems to be a little off
        # translation_matrix = np.array([
        #     [1, 0, 0],
        #     [0, 1, 3]
        # ], dtype=np.float32)
        # h, w = mask.shape[:2]
        # mask_translated = cv2.warpAffine(mask, translation_matrix, (w, h))

        _, thresh = cv2.threshold(
            image[..., 0], 1, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY
        )
        thresh_neg = cv2.bitwise_not(thresh)

        # selected_mask = cv2.bitwise_and(thresh, mask_translated)
        # selected_mask_neg = cv2.bitwise_and(thresh_neg, mask_translated)
        selected_mask = cv2.bitwise_and(thresh, og_mask)
        selected_mask_neg = cv2.bitwise_and(thresh_neg, og_mask)
        selected_mask_dia = cv2.bitwise_and(thresh, mask)
        selected_mask_neg_dia = cv2.bitwise_and(thresh_neg, mask)

        # Choose between white or black section (for white & black text) in each bbox
        for bbox in bboxs:
            x1, y1, x2, y2 = bbox

            roi_sel_mask = selected_mask[y1:y2, x1:x2]
            roi_sel_mask_neg = selected_mask_neg[y1:y2, x1:x2]

            if roi_sel_mask.sum() > roi_sel_mask_neg.sum():
                new_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                    new_mask[y1:y2, x1:x2], selected_mask_dia[y1:y2, x1:x2]
                )
            else:
                new_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                    new_mask[y1:y2, x1:x2], selected_mask_neg_dia[y1:y2, x1:x2]
                )

        new_mask = self.dilate(new_mask, 6)

        return {"og_mask": preds["og_mask"], "mask": new_mask == 1, "bboxs": bboxs}
