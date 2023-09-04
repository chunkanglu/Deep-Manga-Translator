import requests
import os
from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st
from tqdm import tqdm
from zipfile import ZipFile
from io import BytesIO

from deep_translator import GoogleTranslator, DeeplTranslator
from manga_ocr import MangaOcr

from src.inpainter.coarse_gan_inpainter import CoarseGANInpainter
from src.translation import Translation
from src.processor.bubble_seg_processor import BubbleSegProcessor
from src.processor.text_seg_processor import TextSegProcessor
from src.processor.combo_seg_processor import ComboSegProcessor
from src.segmentation.text_seg import TextSegmentationModel, ThresholdTextSegmentationModel
from src.segmentation.pytorch_bubble_seg import PytorchBubbleSegmentationModel

st.set_page_config(layout="wide")

DEVICE = "cpu"
if "ocr" not in st.session_state:
    st.session_state.ocr = MangaOcr(force_cpu=True)
if "downloaded_models" not in st.session_state:
    st.session_state.downloaded_models = False
if "loaded_model" not in st.session_state:
    st.session_state.loaded_model = None

TEXT_SEG_MODEL = "text_seg_model.pth"
BUBBLE_SEG_MODEL = "bubble_seg_model.pth"
COARSE_INPAINT_MODEL = "coarse_gen_states_places2.pth"
FONT = "wildwordsroman.TTF"

def download_models():
    if not os.path.exists("assets"):
        os.makedirs("assets")

    needed_models = [TEXT_SEG_MODEL, BUBBLE_SEG_MODEL, COARSE_INPAINT_MODEL, FONT]
    for i in needed_models:
        model_path = Path("./assets", i)
        if not model_path.exists():
            print(f"Downloading {i}")
            model_url = f"https://github.com/chunkanglu/Manga-Translator/releases/download/v0.1.0/{i}"

            res = requests.get(model_url, stream=True)
            file_size = int(res.headers.get("Content-Length", 0))
            block_size = 1024
            progress_bar = tqdm(total=file_size, unit="iB", unit_scale=True)

            with open(str(model_path), "wb") as f:
                for data in res.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)

            progress_bar.close()

@st.cache_resource(show_spinner=False)
def get_processor():
    tr = DeeplTranslator("ja", "en", api_key=st.secrets["DEEPL_API_KEY"])
    seg_text = ThresholdTextSegmentationModel(f"./assets/{TEXT_SEG_MODEL}", DEVICE)
    seg_bub = PytorchBubbleSegmentationModel(f"./assets/{BUBBLE_SEG_MODEL}", DEVICE)
    ip = CoarseGANInpainter(checkpoint=f"./assets/{COARSE_INPAINT_MODEL}", device=DEVICE)
    return ComboSegProcessor(seg_bub,
                             seg_text,
                             ip,
                             tr,
                             st.session_state.ocr,
                             DEVICE)

def main():
    st.title("Deep Manga Translator")

    st.write("A fully machine Japanese to English translation service for manga panels.")

    if not st.session_state.downloaded_models:
        with st.spinner("Downloading/Loading Model..."):
            download_models()
        st.session_state.downloaded_models = True

        pr = get_processor()
        st.session_state.loaded_model = Translation(pr,
                                                    f"assets/{FONT}")
    else:
        st.warning("Please wait until models have finished downloading")

    with st.form(key="input", clear_on_submit=True):
        n_cols = st.number_input("Number of side-by-side images:", 1, 10, 1)

        image_files = st.file_uploader("Upload Images", type=['png', 'jpeg', 'jpg'], accept_multiple_files=True)

        go = st.form_submit_button("Reset & Go")

    if st.session_state.loaded_model and go:
        if image_files is not None and image_files != []:

            original, translated = st.columns(2)
            original.header("Raw")
            translated.header("Translated")

            n_cols = min(len(image_files), n_cols)

            n_row = int(1 + len(image_files) // int(n_cols))

            image_files = sorted(image_files, key=lambda x: x.name)

            with st.spinner("Processing..."):

                with original:
                    rows1 = [st.columns(int(n_cols)) for _ in range(n_row)]
                    cols1 = [column for row in rows1 for column in row]
                    for col, og in zip(cols1, image_files):
                        col.image(og)

                image_pair = [(image.name, image) for image in image_files]

                zip_file = BytesIO()

                with translated:
                    rows2 = [st.columns(int(n_cols)) for _ in range(n_row)]
                    cols2 = [column for row in rows2 for column in row]

                    with ZipFile(zip_file, 'w') as zf:
                        for col, (image_name, image) in zip(cols2, image_pair):
                            image = Image.open(image)
                            trans = st.session_state.loaded_model.translate_page(image)

                            img_obj = BytesIO()
                            trans.save(img_obj, "PNG")

                            new_name = image_name[:-4] + "_translated.png"
                            zf.writestr(new_name, img_obj.getvalue())
                            col.image(trans)

                            trans.close()

            st.download_button("Download output & reset",
                              data=zip_file,
                              file_name="output.zip",
                              mime="application/zip")


if __name__ == "__main__":
    main()
