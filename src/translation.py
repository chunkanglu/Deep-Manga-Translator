import cv2
import numpy as np
# import largestinteriorrectangle as lir
# import os
# import re
# from pathlib import Path
# from requests import get
# from tqdm import tqdm

# from detectron2 import model_zoo
# from detectron2.config import get_cfg
# from detectron2.engine import DefaultPredictor
# import torch

# from deep_translator import GoogleTranslator, DeeplTranslator
# from manga_ocr import MangaOcr
from PIL import Image, ImageFont, ImageDraw
from src.processor.baseprocessor import BaseProcessor

# from typing import Any, Optional
import numpy.typing as npt


# def download_model(seg_model_path: str):
#     model_url = "https://github.com/chunkanglu/Manga-Translator/releases/download/v0.1.0/model.pth"

#     res = get(model_url, stream=True)
#     file_size = int(res.headers.get("Content-Length", 0))
#     block_size = 1024
#     progress_bar = tqdm(total=file_size, unit="iB", unit_scale=True)

#     with open(seg_model_path, "wb") as f:
#         for data in res.iter_content(block_size):
#             progress_bar.update(len(data))
#             f.write(data)

#     progress_bar.close()


# def download_font(font: str):
#     font_url = "https://github.com/chunkanglu/Manga-Translator/releases/download/v0.1.0/wildwordsroman.TTF"

#     res = get(font_url, stream=True)
#     file_size = int(res.headers.get("Content-Length", 0))
#     block_size = 1024
#     progress_bar = tqdm(total=file_size, unit="iB", unit_scale=True)

#     with open(font, "wb") as f:
#         for data in res.iter_content(block_size):
#             progress_bar.update(len(data))
#             f.write(data)


class Translation:
    def __init__(self,
                 processor: BaseProcessor,
                 font: str = "assets/wildwordsroman.TTF",
                 ) -> None:
    # def __init__(self,
    #              processor: BaseProcessor,
    #              seg_model_path: str = "assets/model.pth",
    #              translator: str = "Deepl",
    #              src: str = "ja",
    #              tgt: str = "en",
    #              text_buffer: float = 0.95,
    #              font: str = "assets/wildwordsroman.TTF",
    #              api_key: Optional[str] = None,
    #              ) -> None:

        # model_path = Path(seg_model_path)
        # font_path = Path(font)

        # if not os.path.exists("assets"):
        #     os.makedirs("assets")

        # if not model_path.exists():
        #     download_model(seg_model_path)

        # if not font_path.exists():
        #     download_font(font)

        # if (src == "ja"):
        #     self.ocr = MangaOcr()

        # if translator == "Deepl":
        #     self.tr = DeeplTranslator(api_key=api_key, source=src, target=tgt)
        # elif translator == "Google":
        #     self.tr = GoogleTranslator(source=src, target=tgt)
        # else:
        #     raise Exception("Invalid Translator")
        
        self.processor = processor

        self.font = font

    def process_image(self,
                      image
                      ) -> npt.NDArray[np.uint8]:
        image = np.asarray(image, dtype=np.uint8)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return image.astype(np.uint8)
    
    def translate_page(self,
                       image
                       ) -> Image.Image:
        if isinstance(image, str):
            image = Image.open(image)
        image = self.process_image(image)
        clean_image = self.processor.clean_text(image)
        output_image = self.processor.add_translated_text(image,
                                                          clean_image,
                                                          self.font)
        
        return output_image


