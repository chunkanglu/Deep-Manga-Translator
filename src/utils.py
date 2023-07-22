import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
import re

def get_crop(img: npt.NDArray[np.uint8],
             bbox: tuple[int, int, int, int]
             ) -> npt.NDArray[np.uint8]:
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2]

def get_text(img: npt.NDArray[np.uint8],
             ocr_model
             ) -> str:
    return ocr_model(Image.fromarray(img))

def get_tr_text(text: str,
                translator
                ) -> str:
    return translator.translate(text)

def draw_text(bbox: tuple[int, int, int, int],
              tr_text: str,
              img_to_draw: ImageDraw.ImageDraw,
              font_path: str,
              text_buffer: float = 0.95,
              ) -> None:
    x1, y1, x2, y2 = bbox
    mid_v = (x1 + x2) // 2
    mid_h = (y1 + y2) // 2
    max_buffer_x = int((x2 - x1) * text_buffer)
    max_buffer_y = int((y2 - y1) * text_buffer)

    if tr_text is None:
        return

    text_arr = re.split(r'[\s\-]', tr_text)

    font_size = 200

    while True:

        curr_font = ImageFont.truetype(font_path, font_size)

        multi_line = "\n"
        next_line = ""

        for t in text_arr:

            while (img_to_draw.textlength(t,
                                            font=curr_font) >= max_buffer_x):
                font_size -= 2
                curr_font = ImageFont.truetype(font_path, font_size)


            if (img_to_draw.textlength(next_line + " " + t,
                                        font=curr_font) < max_buffer_x):
                if (next_line == ""):
                    next_line = t
                else:
                    next_line = next_line + " " + t

            elif (img_to_draw.textlength(next_line,
                                            font=curr_font) < max_buffer_x):
                multi_line += next_line + "\n"
                next_line = t

        multi_line += next_line + "\n"

        _, top, _, bottom = img_to_draw.multiline_textbbox((mid_v, mid_h),
                                                            multi_line,
                                                            font=curr_font)

        if (bottom - top < max_buffer_y):
            break

        font_size -= 2

    img_to_draw.multiline_text((mid_v, mid_h),
                                multi_line,
                                (0, 0, 0),
                                font=curr_font,
                                anchor="mm",
                                align="center")