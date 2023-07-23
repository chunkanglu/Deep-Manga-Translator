import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
import re
from string import punctuation
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


def process_tr_text(text: str | None
                    ) -> str | None:
    if text is None:
        return None

    # Remove leading ...
    text_arr = text.split()
    if text_arr[0] == "..." or text_arr[0] == "..":
        text = " ".join(text_arr[1:])

    # Remove leading & trailing punctuation and spaces
    text = text.strip(punctuation + " ")

    return text


def draw_text(bbox: tuple[int, int, int, int],
              tr_text: str | None,
              img_to_draw: ImageDraw.ImageDraw,
              font_path: str,
              text_buffer: float = 1.0,
              ) -> None:
    x1, y1, x2, y2 = bbox
    mid_v = (x1 + x2) // 2
    mid_h = (y1 + y2) // 2
    max_buffer_x = int((x2 - x1) * text_buffer)
    max_buffer_y = int((y2 - y1) * text_buffer)

    tr_text = process_tr_text(tr_text)

    if (tr_text is None) or (tr_text == ""):
        return

    much_taller_than_wide = (y2 - y1) > (2 * (x2 - x1))
    is_single_word = len(tr_text.split()) == 1

    spacing = 1 if is_single_word and much_taller_than_wide else 4

    print_text = ""

    upper_font_size = 200
    lower_font_size = 1

    while upper_font_size - lower_font_size > 1:
        font_size = (upper_font_size + lower_font_size) // 2

        curr_font = ImageFont.truetype(font_path, font_size)

        # W is the widest character
        max_char_len = int(img_to_draw.textlength("W", font=curr_font))
        row_max_chars = max_buffer_x // max_char_len

        # Skip if even a single char can't fit
        if row_max_chars == 0:
            upper_font_size = font_size
            continue

        if is_single_word and much_taller_than_wide:
            print_text = "\n".join(list(tr_text))
        else:
            print_text = fill(text=tr_text,
                              width=row_max_chars,
                              break_long_words=False,
                              break_on_hyphens=True)

        left, top, right, bottom = img_to_draw.multiline_textbbox((mid_v, mid_h),
                                                                  print_text,
                                                                  font=curr_font,
                                                                  anchor="mm",
                                                                  align="center",
                                                                  spacing=spacing,
                                                                  stroke_width=3)

        if (right - left < max_buffer_x) and (bottom - top < max_buffer_y):
            lower_font_size = font_size
        else:
            upper_font_size = font_size - 1

    print_font = ImageFont.truetype(font_path, lower_font_size)

    img_to_draw.multiline_text((mid_v, mid_h),
                               print_text,
                               (0, 0, 0),
                               font=print_font,
                               anchor="mm",
                               align="center",
                               spacing=spacing,
                               stroke_fill=COLOR_WHITE)
