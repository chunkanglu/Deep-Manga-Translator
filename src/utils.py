import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
import re
from textwrap import fill

COLOR_WHITE = (255, 255, 255)

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

    font_size = 200

    while font_size > 0:

        curr_font = ImageFont.truetype(font_path, font_size)

        # W is the widest character
        max_char_len = int(img_to_draw.textlength("W", font=curr_font))
        row_max_chars = max_buffer_x // max_char_len

        if row_max_chars == 0:
            font_size -= 2
            continue

        multi_line = fill(text=tr_text,
                          width=row_max_chars,
                          break_long_words=False,
                          break_on_hyphens=True)

        left, top, right, bottom = img_to_draw.multiline_textbbox((mid_v, mid_h),
                                                            multi_line,
                                                            font=curr_font)

        if (right - left < max_buffer_x) and (bottom - top < max_buffer_y):
            break

        font_size -= 2

    img_to_draw.multiline_text((mid_v, mid_h),
                                multi_line,
                                (0, 0, 0),
                                font=curr_font,
                                anchor="mm",
                                align="center",
                                stroke_width=3,
                                stroke_fill=COLOR_WHITE)