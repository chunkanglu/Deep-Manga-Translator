import largestinteriorrectangle as lir
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
import re

from src.processor.baseprocessor import BaseProcessor
from src.segmentation.bubble_seg import BubbleSegmentationModel
from src.utils import get_crop, get_text, get_tr_text, draw_text

class BubbleSegProcessor(BaseProcessor):
    def __init__(self,
                 seg_model: BubbleSegmentationModel,
                 inpaint_model,
                 translator,
                 ocr_model) -> None:
        super().__init__(seg_model, inpaint_model, translator, ocr_model)

    # def get_masks(self,
    #               image: npt.NDArray[np.uint8]
    #               ) -> list[npt.NDArray[np.bool_]]:
    #     preds = self.seg_model.predict(image)
    #     return preds["masks"]
    
    # def get_text_bboxs(self,
    #                    image: npt.NDArray[np.uint8]
    #                    ) -> list[tuple[int, int, int, int]]:
    #     preds = self.seg_model.predict(image)
    #     return preds["bboxs"]
    
    # TODO: use inpainting here
    def clean_text(self,
                   image: npt.NDArray[np.uint8]
                   ) -> npt.NDArray[np.uint8]:
        preds = self.seg_model.predict(image)
        image = image.copy()
        for mask in preds["masks"]:
            image[mask, :] = [255, 255, 255]
        return image

    def add_translated_text(self,
                            image: npt.NDArray[np.uint8],
                            clean_image: npt.NDArray[np.uint8],
                            font_path: str
                            ) -> Image.Image:
        
        def get_largest_text_box(mask: npt.NDArray[np.bool_]
                                    ) -> npt.NDArray[np.uint32]:
                return lir.lir(mask).astype(np.uint32)

        # def draw_text(mask: npt.NDArray[np.bool_],
        #               tr_text: str,
        #               img_to_draw: ImageDraw.ImageDraw,
        #               font_path: str,
        #               text_buffer: float = 0.95,
        #               ) -> None:
            
            

        #     (x, y, w, h) = get_largest_text_box(mask)
        #     mid_v = x + w // 2
        #     mid_h = y + h // 2
        #     max_buffer_x = int(w * text_buffer)
        #     max_buffer_y = int(h * text_buffer)

        #     if tr_text is None:
        #         return

        #     text_arr = re.split(r'[\s\-]', tr_text)

        #     font_size = 200

        #     while True:

        #         curr_font = ImageFont.truetype(font_path, font_size)

        #         multi_line = "\n"
        #         next_line = ""

        #         for t in text_arr:

        #             while (img_to_draw.textlength(t,
        #                                           font=curr_font) >= max_buffer_x):
        #                 font_size -= 2
        #                 curr_font = ImageFont.truetype(font_path, font_size)


        #             if (img_to_draw.textlength(next_line + " " + t,
        #                                        font=curr_font) < max_buffer_x):
        #                 if (next_line == ""):
        #                     next_line = t
        #                 else:
        #                     next_line = next_line + " " + t

        #             elif (img_to_draw.textlength(next_line,
        #                                          font=curr_font) < max_buffer_x):
        #                 multi_line += next_line + "\n"
        #                 next_line = t

        #         multi_line += next_line + "\n"

        #         _, top, _, bottom = img_to_draw.multiline_textbbox((mid_v, mid_h),
        #                                                            multi_line,
        #                                                            font=curr_font)

        #         if (bottom - top < max_buffer_y):
        #             break

        #         font_size -= 2

        #     img_to_draw.multiline_text((mid_v, mid_h),
        #                                multi_line,
        #                                (0, 0, 0),
        #                                font=curr_font,
        #                                anchor="mm",
        #                                align="center")

        output_image = Image.fromarray(clean_image.copy())
        draw = ImageDraw.Draw(output_image)

        preds = self.seg_model.predict(image)

        for mask, bbox in zip(preds["masks"], preds["bboxs"]):
            output_image = Image.fromarray(output_image)
            draw = ImageDraw.Draw(output_image)

            crop = get_crop(image, bbox)

            og_text = get_text(crop, self.ocr_model)
            tr_text = get_tr_text(og_text, self.translator)

            x, y, w, h = get_largest_text_box(mask)

            draw_text((x, y, x+w, y+h), tr_text, draw, font_path)

        return output_image