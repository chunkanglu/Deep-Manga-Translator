import cv2
import numpy as np
import largestinteriorrectangle as lir
import os
import re

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from deep_translator import GoogleTranslator
from manga_ocr import MangaOcr
from PIL import Image, ImageFont, ImageDraw

class Translation:
    def __init__(self, 
                 src="ja", 
                 tgt="en", 
                 seg_model_path="assets\model.pth",
                 text_buffer=0.9,
                 font="assets\wildwordsroman.TTF") -> None:
        if (src == "ja"):
            self.ocr = MangaOcr()
        self.tr = GoogleTranslator(source=src, target=tgt)

        seg_model_head, seg_model_tail = os.path.split(seg_model_path)
        cfg_pred = get_cfg()
        cfg_pred.OUTPUT_DIR = seg_model_head
        cfg_pred.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg_pred.MODEL.WEIGHTS = os.path.join(cfg_pred.OUTPUT_DIR, seg_model_tail)
        cfg_pred.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9

        self.predictor = DefaultPredictor(cfg_pred)

        self.text_buffer = text_buffer
        self.font = font

    def read_img(self, img_path):
        img_t = Image.open(img_path)
        img = np.array(img_t)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img.copy()

    def predict(self, img):
        return self.predictor(img)["instances"].to("cpu").get_fields()

    def clean_text_boxes(self, img, prediction):
        img_copy = img.copy()
        for mask in prediction["pred_masks"].numpy():
            img_copy[mask, :] = [255, 255, 255]
        return img_copy

    def get_cropped_bboxs(self, img, prediction):
        bboxs = prediction["pred_boxes"].tensor.numpy().astype(int)

        cropped = []
        for b in bboxs:
            x1, y1, x2, y2 = b
            crop = img[y1:y2, x1:x2]
            cropped.append(crop)

        return cropped

    def get_largest_text_box(self, mask):
        return lir.lir(mask.numpy().astype(bool))

    def get_text_array(self, bboxs):
        text = []
        for i in bboxs:
            text.append(self.ocr(Image.fromarray(i)))
        return text

    def get_translated_text(self, text_array):
        tr_text = []
        for i in text_array:
            tr_text.append(self.tr.translate(i))
        print(text_array)
        print(tr_text)
        return tr_text

    def translate(self, img_path):
        img = self.read_img(img_path)
        preds = self.predict(img)

        clean_img = self.clean_text_boxes(img, preds)
        bboxs = self.get_cropped_bboxs(img, preds)

        og_text = self.get_text_array(bboxs)
        translated_text = self.get_translated_text(og_text)

        output_img = Image.fromarray(clean_img)
        draw = ImageDraw.Draw(output_img)

        for mask, tr_text in zip(preds["pred_masks"], translated_text):

            (x, y, w, h) = self.get_largest_text_box(mask)
            mid_v = x + w // 2
            mid_h = y + h // 2
            maxBuffer = int(w * self.text_buffer)
            font_size = 200

            if(tr_text == None):
                continue

            text_arr = re.split(r'[\s\-]', tr_text)
            multi_line = "\n"
            next_line = ""

            while True:

                multi_line = "\n"
                next_line = ""

                for t in text_arr:

                    while (draw.textlength(t, font=ImageFont.truetype(self.font, font_size)) >= maxBuffer):
                        font_size -= 2

                    if (draw.textlength(next_line + " " + t, font=ImageFont.truetype(self.font, font_size)) < maxBuffer):
                        if (next_line == ""):
                            next_line = t
                        else:
                            next_line = next_line + " " + t

                    elif (draw.textlength(next_line, font=ImageFont.truetype(self.font, font_size)) < maxBuffer):
                        multi_line += next_line + "\n"
                        next_line = t

                multi_line += next_line + "\n"

                left, top, right, bottom = draw.multiline_textbbox((mid_v, mid_h), multi_line, font=ImageFont.truetype(self.font, font_size))

                if (bottom-top < h):
                    break

                font_size -= 2

            draw.multiline_text((mid_v, mid_h), multi_line, (0,0,0), font=ImageFont.truetype(self.font, font_size), anchor="mm", align="center")

        return output_img
