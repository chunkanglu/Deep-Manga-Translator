from src.translation import Translation
from  src.processor.text_seg_processor import TextSegProcessor
from src.segmentation.text_seg import TextSegmentationModel
import cv2
import matplotlib.pyplot as plt

from deep_translator import GoogleTranslator, DeeplTranslator
from manga_ocr import MangaOcr

from PIL import Image
from dotenv import load_dotenv
import os

def main():
    load_dotenv()

    # processor = 
    # tr = Translation(translator="Google")
    # img_path = r"C:\Users\Kang\Documents\Machine Learning\Manga_thing\manga_bubbles_test_1\images\10482900-Densetsu_no_ryuusou_14_1.jpg"
    # out_path = r"res.png"

    # output_img = tr.translate(img_path)

    # output_img.show()
    # output_img.save(out_path)

    

    seg = TextSegmentationModel("/home/chunkanglu/Documents/Deep_Manga_Translator/experiments/best_model_v2.pth",
                                "cpu")
    # translator = GoogleTranslator("ja", "en")
    translator = DeeplTranslator("ja", "en", os.environ.get("DEEPL_API_KEY"))
    ocr = MangaOcr()
    processor = TextSegProcessor(seg,
                                 None,
                                 translator,
                                 ocr)
    
    tr = Translation(processor)
    
    output = tr.translate_page("experiments/manga_bubbles_train_2/images/f6dbb986-Daring_in_the_Franxx_44_4.jpg")
    output.show()

if __name__ == "__main__":
    main()
