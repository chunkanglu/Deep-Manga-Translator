import streamlit as st
from PIL import Image
from src.translation import Translation


@st.cache_resource
def translation_model():
    return Translation()


def main():
    st.title("Deep Manga Translator")

    tr = translation_model()
    image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])

    if image_file is not None:
        img = Image.open(image_file)

        st.image(img, caption='Uploaded Image.')

        with st.spinner("Processing..."):
            img_ = tr.translate(image_file)
        st.image(img_, caption='Proccesed Image.')


if __name__ == "__main__":
    main()
