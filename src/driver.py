from translation import Translation
from  processor.text_seg_processor import TextSegProcessor
from segmentation.text_seg import TextSegmentationModel
import cv2
import matplotlib.pyplot as plt

from deep_translator import GoogleTranslator, DeeplTranslator
from manga_ocr import MangaOcr

from PIL import Image

def main():

    # processor = 
    # tr = Translation(translator="Google")
    # img_path = r"C:\Users\Kang\Documents\Machine Learning\Manga_thing\manga_bubbles_test_1\images\10482900-Densetsu_no_ryuusou_14_1.jpg"
    # out_path = r"res.png"

    # output_img = tr.translate(img_path)

    # output_img.show()
    # output_img.save(out_path)

    

    seg = TextSegmentationModel("/home/chunkanglu/Documents/Deep_Manga_Translator/experiments/best_model_v2.pth",
                                "cpu")
    translator = GoogleTranslator("ja", "en")
    ocr = MangaOcr()
    processor = TextSegProcessor(seg,
                                 None,
                                 translator,
                                 ocr)
    
    tr = Translation(processor)
    
    output = tr.translate_page("/home/chunkanglu/Documents/Deep_Manga_Translator/experiments/MangaDatasetSeparate/images/e485d130-boukensha_guild_17_3.jpg")
    output.show()

    output = tr.translate_page("/home/chunkanglu/Documents/Deep_Manga_Translator/experiments/DATA/Manga109s_released_2021_12_30/images/Arisa/000.jpg")
    output.show()

if __name__ == "__main__":
    main()
