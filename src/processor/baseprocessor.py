from PIL import Image
import numpy as np
import numpy.typing as npt

from segmentation.basemodel import BaseModel

class BaseProcessor():
    def __init__(self,
                 seg_model: BaseModel,
                 inpaint_model,
                 translator,
                 ocr_model) -> None:
        self.seg_model = seg_model
        self.inpaint_model = inpaint_model
        self.translator = translator
        self.ocr_model = ocr_model

        self.last_image = None
        self.prediction = None

    # def get_masks(self,
    #               image: npt.NDArray[np.uint8]
    #               ) -> list[npt.NDArray[np.bool_]]:
    #     raise NotImplementedError
    
    # def get_text_bboxs(self,
    #                    image: npt.NDArray[np.uint8]
    #                    ) -> list[tuple[int, int, int, int]]:
    #     raise NotImplementedError

    # def empty_cache(self) -> None:
    #     self.last_image = None

    def clean_text(self,
                   image: npt.NDArray[np.uint8]
                   ) -> npt.NDArray[np.uint8]:
        raise NotImplementedError
    
    def add_translated_text(self,
                            image: npt.NDArray[np.uint8],
                            clean_image: npt.NDArray[np.uint8],
                            font_path: str
                            ) -> Image.Image:
        raise NotImplementedError