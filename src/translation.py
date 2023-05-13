import cv2
import numpy as np
import largestinteriorrectangle as lir
import os
import re
from pathlib import Path
from requests import get
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import torch

from deep_translator import GoogleTranslator, DeeplTranslator
from manga_ocr import MangaOcr
from PIL import Image, ImageFont, ImageDraw

from typing import Any, Optional
import numpy.typing as npt


def download_model(seg_model_path: str):
    model_url = "https://github.com/chunkanglu/Manga-Translator/releases/download/v0.1.0/model.pth"

    res = get(model_url, stream=True)
    file_size = int(res.headers.get("Content-Length", 0))
    block_size = 1024
    progress_bar = tqdm(total=file_size, unit="iB", unit_scale=True)

    with open(seg_model_path, "wb") as f:
        for data in res.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)

    progress_bar.close()


def download_font(font: str):
    font_url = "https://github.com/chunkanglu/Manga-Translator/releases/download/v0.1.0/wildwordsroman.TTF"

    res = get(font_url, stream=True)
    file_size = int(res.headers.get("Content-Length", 0))
    block_size = 1024
    progress_bar = tqdm(total=file_size, unit="iB", unit_scale=True)

    with open(font, "wb") as f:
        for data in res.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)


class Translation:
    def __init__(self,
                 translator: str = "Deepl",
                 src: str = "ja",
                 tgt: str = "en",
                 seg_model_path: str = "assets/model.pth",
                 text_buffer: float = 0.9,
                 font: str = "assets/wildwordsroman.TTF",
                 api_key: Optional[str] = None,
                 ) -> None:

        model_path = Path(seg_model_path)
        font_path = Path(font)

        if not model_path.exists():
            download_model(seg_model_path)

        if not font_path.exists():
            download_font(font)

        if (src == "ja"):
            self.ocr = MangaOcr()

        if translator == "Deepl":
            self.tr = DeeplTranslator(api_key=api_key, source=src, target=tgt)
        elif translator == "Google":
            self.tr = GoogleTranslator(source=src, target=tgt)
        else:
            raise Exception("Invalid Translator")

        seg_model_head, seg_model_tail = os.path.split(seg_model_path)
        cfg_pred = get_cfg()
        if not torch.cuda.is_available():
            cfg_pred.MODEL.DEVICE = "cpu"
        cfg_pred.OUTPUT_DIR = seg_model_head
        cfg_pred.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg_pred.MODEL.WEIGHTS = os.path.join(cfg_pred.OUTPUT_DIR,
                                              seg_model_tail)
        cfg_pred.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9

        self.predictor = DefaultPredictor(cfg_pred)

        self.text_buffer = text_buffer
        self.font = font

    def read_img(self,
                 img_path: str
                 ) -> npt.NDArray[np.uint8]:
        img_t = Image.open(img_path)
        img = np.asarray(img_t, dtype=np.uint8)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img.astype(np.uint8)

    def predict(self,
                img: npt.NDArray[np.uint8]
                ) -> dict[str, Any]:
        return self.predictor(img)["instances"].to("cpu").get_fields()

    def get_largest_text_box(self,
                             mask: npt.NDArray[np.bool_]
                             ) -> npt.NDArray[np.uint32]:
        return lir.lir(mask).astype(np.uint32)

    def clean_text_box(self,
                       img: npt.NDArray[np.uint8],
                       mask: npt.NDArray[np.bool_]
                       ) -> npt.NDArray[np.uint8]:
        img_copy = img.copy()
        img_copy[mask, :] = [255, 255, 255]
        return img_copy

    def get_crop(self,
                 img:npt.NDArray[np.uint8],
                 bbox:
                 tuple[int, int, int, int]
                 ) -> npt.NDArray[np.uint8]:
        x1, y1, x2, y2 = bbox
        return img[y1:y2, x1:x2]

    def get_text(self,
                 img: npt.NDArray[np.uint8]
                 ) -> str:
        return self.ocr(Image.fromarray(img))

    def get_tr_text(self,
                    text: str
                    ) -> str:
        return self.tr.translate(text)

    def draw_text(self,
                  mask: npt.NDArray[np.bool_],
                  tr_text: str,
                  img_to_draw: ImageDraw.ImageDraw
                  ) -> None:
        (x, y, w, h) = self.get_largest_text_box(mask)
        mid_v = x + w // 2
        mid_h = y + h // 2
        max_buffer_x = int(w * self.text_buffer)
        max_buffer_y = int(h * self.text_buffer)

        if tr_text is None:
            return

        text_arr = re.split(r'[\s\-]', tr_text)

        font_size = 200

        while True:

            curr_font = ImageFont.truetype(self.font, font_size)

            multi_line = "\n"
            next_line = ""

            for t in text_arr:

                while (img_to_draw.textlength(t,
                                              font=curr_font) >= max_buffer_x):
                    font_size -= 2
                    curr_font = ImageFont.truetype(self.font, font_size)


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

    def translate(self,
                  img_path: str
                  ) -> Image.Image:
        img = self.read_img(img_path)
        output_img = img.copy()
        preds = self.predict(img)

        masks = preds["pred_masks"].numpy().astype(bool)
        bboxs = preds["pred_boxes"].tensor.numpy().astype(int)

        for mask, bbox in zip(masks, bboxs):
            output_img = self.clean_text_box(np.asarray(output_img), mask)
            output_img = Image.fromarray(output_img)
            draw = ImageDraw.Draw(output_img)

            crop = self.get_crop(img, bbox)

            og_text = self.get_text(crop)
            tr_text = self.get_tr_text(og_text)

            self.draw_text(mask, tr_text, draw)

        return output_img
