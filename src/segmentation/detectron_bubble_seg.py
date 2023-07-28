import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import torch

import numpy as np
import numpy.typing as npt
from typing import Any

from src.segmentation.basemodel import BaseModel

class Detectron2BubbleSegmentationModel(BaseModel):
    def __init__(self,
                 model_path: str,
                 device: str = "cpu") -> None:
        super().__init__(model_path, device)

        seg_model_head, seg_model_tail = os.path.split(model_path)
        cfg_pred = get_cfg()
        cfg_pred.MODEL.DEVICE = self.device
        cfg_pred.OUTPUT_DIR = seg_model_head
        cfg_pred.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg_pred.MODEL.WEIGHTS = os.path.join(cfg_pred.OUTPUT_DIR,
                                              seg_model_tail)
        cfg_pred.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9

        self.predictor = DefaultPredictor(cfg_pred)

    def predict(self,
                image: npt.NDArray[np.uint8]
                ) -> dict[str, Any]:
        preds = self.predictor(image)["instances"].to("cpu").get_fields()
        masks = preds["pred_masks"].numpy().astype(bool)
        bboxs = preds["pred_boxes"].tensor.numpy().astype(int)
        return {
            "masks": masks,
            "bboxs": bboxs
        }
