from enum import Enum
import largestinteriorrectangle as lir
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
from textwrap import fill

from typing import Union

COLOR_WHITE = (255, 255, 255)
TEXT_BUFFER = 0.95


class DeviceEnum(Enum):
    CPU = "cpu"
    CUDA = "cuda"


def get_crop(
    img: npt.NDArray[np.uint8], bbox: tuple[int, int, int, int]
) -> npt.NDArray[np.uint8]:
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2]


def get_text(img: npt.NDArray[np.uint8], ocr_model) -> str:
    return ocr_model(Image.fromarray(img))


def get_tr_text(text: str, translator) -> str:
    return translator.translate(text)


def process_ocr_text(text: str) -> str:
    return text.replace("．", ".")


def process_tr_text(text: Union[str, None]) -> Union[str, None]:
    if text is None:
        return None

    # Remove leading & trailing spaces
    text = text.strip()
    return text


def ocr_bbox_sort(d):
    # Right to left, top to bottom using centre of bbox
    _, bbox = d
    x, y, w, h = bbox
    return (y + h // 2, -x - w // 2)


def get_text_box(bbox: tuple[int, int, int, int], mask: npt.NDArray[np.bool_]) -> tuple[int, int, int, int]:
    # First try to expand text bbox
    if expanded_box := expand_text_box(bbox=bbox, mask=mask):
        return expanded_box
    # If that fails (ie. bbox sticks out of bubble mask), fall back to largest interior rectangle
    return get_largest_text_box(mask=mask)

def expand_text_box(
    bbox: tuple[int, int, int, int], mask: npt.NDArray[np.bool_]
) -> tuple[int, int, int, int] | None:
    """
    Enlarges text bounding box horizontally to edge of mask.

    Gets the left and right distance from bbox left and
    right vertical edge and expands symmetrically to the
    smaller value to ensure box is still centered.
    """
    x1, y1, x2, y2 = bbox

    mask_crop_right = mask[y1:y2, x2:]
    mask_crop_left = mask[y1:y2, :x1]

    right_expanded = x2
    left_expanded = x1

    try:
        right_mask_bound = int(np.min(np.where(~mask_crop_right)[1]))
        right_expanded = max(right_expanded, x2 + right_mask_bound)
    except ValueError:
        print("Cannot find right edge of mask.")
        return None

    try:
        left_mask_bound = int(np.max(np.where(~mask_crop_left)[1]))
        left_expanded = min(left_expanded, left_mask_bound)
    except ValueError:
        print("Cannot find left edge of mask.")
        return None

    return (left_expanded, y1, right_expanded, y2)


def get_largest_text_box(mask: npt.NDArray[np.bool_]) -> tuple[int, int, int, int]:
    x, y, w, h = lir.lir(mask).astype(np.uint32)
    return (x, y, x+w, y+h)


def draw_text(
    bbox: tuple[int, int, int, int],
    tr_text: Union[str, None],
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

    text_chunks = tr_text.split()

    print_text = ""

    upper_font_size = 100
    lower_font_size = 5

    while upper_font_size - lower_font_size > 1:
        font_size = (upper_font_size + lower_font_size) // 2

        curr_font = ImageFont.truetype(font_path, font_size)

        lines = []
        line = ""
        for word in text_chunks:
            new_line = line + " " + word
            l, t, r, b = img_to_draw.textbbox(
                (mid_v, mid_h),
                new_line,
                font=curr_font,
                anchor="mm",
                align="center",
                spacing=1,
                stroke_width=3,
            )
            if r - l <= max_buffer_x:
                line = new_line
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)

        print_text = "\n".join(lines)

        left, top, right, bottom = img_to_draw.multiline_textbbox(
            (mid_v, mid_h),
            print_text,
            font=curr_font,
            anchor="mm",
            align="center",
            spacing=1,
            stroke_width=3,
        )

        if (right - left < max_buffer_x) and (bottom - top < max_buffer_y):
            lower_font_size = font_size
        else:
            upper_font_size = font_size - 1

    print_font = ImageFont.truetype(font_path, lower_font_size)

    img_to_draw.multiline_text(
        (mid_v, mid_h),
        print_text,
        (0, 0, 0),
        font=print_font,
        anchor="mm",
        align="center",
        spacing=1,
        stroke_width=3,
        stroke_fill=COLOR_WHITE,
    )
