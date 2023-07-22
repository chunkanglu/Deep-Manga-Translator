import albumentations as A
import cv2
import numpy as np
import numpy.typing as npt
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch
from typing import Any


from segmentation.basemodel import BaseModel

class TextSegmentationModel(BaseModel):
    def __init__(self,
                 model_path: str,
                 device: str
                 ) -> None:
        super().__init__(model_path, device)

        def to_tensor(x, **kwargs):
            return x.transpose(2, 0, 1).astype('float32')

        def to_tensor_mask(x, **kwargs):
            return np.expand_dims(x, axis=0).astype('float32')

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

        preprocessing_fn = get_preprocessing_fn(encoder_name="resnet34",
                                                pretrained="imagenet")
        
        self.preprocessing = get_preprocessing(preprocessing_fn)

        self.dialate = lambda x: cv2.morphologyEx(x,
                                                  cv2.MORPH_DILATE,
                                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                            (12, 12)))
        
        self.predictor = torch.load(self.model_path).to(self.device)

    def predict(self,
                image: npt.NDArray[np.uint8]
                ) -> dict[str, Any]:
        og_shape = image.shape
        padded_height = og_shape[0] + 32 - (og_shape[0] % 32)
        padded_width = og_shape[1] + 32 - (og_shape[1] % 32)

        image = self.preprocessing(image=image)["image"]
        image = np.transpose(image, (1, 2, 0))
        image = A.PadIfNeeded(padded_height,
                              padded_width,
                              position=A.PadIfNeeded.PositionType.TOP_LEFT)(image=image)["image"]

        image_tensor = torch.from_numpy(image).to(self.device).unsqueeze(0)

        prediction = self.predictor.predict(image_tensor.permute(0, 3, 1, 2).float())

        pred_mask = prediction.squeeze().cpu().numpy()
        pred_mask = pred_mask[:og_shape[0], :og_shape[1]]
        pred_mask = self.dialate(pred_mask)
        dialated = self.dialate(pred_mask)

        pred_mask = pred_mask > 0.5

        thresh = cv2.threshold(dialated, 0.5, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh,
                                                                    connectivity=8)
        bboxes = []
        for label in range(1, num_labels):
            x1 = stats[label, cv2.CC_STAT_LEFT]
            y1 = stats[label, cv2.CC_STAT_TOP]
            x2 = x1 + stats[label, cv2.CC_STAT_WIDTH]
            y2 = y1 + stats[label, cv2.CC_STAT_HEIGHT]

            bboxes.append((x1, y1, x2, y2))

        return {
            "mask": pred_mask,
            "bboxs": bboxes
        }

