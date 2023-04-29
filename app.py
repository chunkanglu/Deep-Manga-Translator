import streamlit as st
from src.translation import Translation
from zipfile import ZipFile
from io import BytesIO

st.set_page_config(layout="wide")


@st.cache_resource
def translation_model():
    return Translation()


def main():
    st.title("Deep Manga Translator")

    st.write("A fully machine Japanese to English translation service for manga panels.")

    with st.spinner("Downloading/Loading Model..."):
        tr = translation_model()

    with st.form(key="input"):
        n_cols = st.number_input("Number of side-by-side images:", 1, 10, 1)

        image_files = st.file_uploader("Upload Images", type=['png', 'jpeg', 'jpg'], accept_multiple_files=True)

        go = st.form_submit_button("Reset & Go")

    if go:
        if image_files != []:

            original, translated = st.columns(2)
            original.header("Raw")
            translated.header("Translated")

            n_cols = min(len(image_files), n_cols)

            n_row = int(1 + len(image_files) // int(n_cols))

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
                            trans = tr.translate(image)

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
