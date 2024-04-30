import numpy as np
import numpy.typing as npt
import torch
from torchvision.transforms import functional as TF
from typing import Any

from src.segmentation.basemodel import BaseModel


class PytorchBubbleSegmentationModel(BaseModel):
    def __init__(self, model_path: str, device: str) -> None:
        super().__init__(model_path, device)

        self.predictor = torch.load(self.model_path, map_location=device).to(device)

    def predict(self, image: npt.NDArray[np.uint8]) -> dict[str, Any]:
        self.predictor.eval()
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.convert_image_dtype(image_tensor, torch.float)
        with torch.no_grad():
            prediction = self.predictor([image_tensor.to(self.device)])

        masks = [
            prediction[0]["masks"][i, 0].mul(100).byte().cpu().numpy() > 95
            for i in range(prediction[0]["masks"].size()[0])
        ]

        bboxs = [
            b.cpu().numpy().astype(np.int64).tolist() for b in prediction[0]["boxes"]
        ]

        OVERLAP_PERCENTAGE = 0.1
        combined_mask = np.zeros((image.shape[0], image.shape[1]))
        i = 0
        while i < len(masks):
            m = masks[i]
            overlap = np.logical_and(combined_mask, m)
            if overlap.sum() >= m.sum() * OVERLAP_PERCENTAGE:
                masks.pop(i)
                bboxs.pop(i)
            else:
                combined_mask = np.logical_or(combined_mask, m)
                i += 1

        return {"masks": masks, "bboxs": bboxs}
