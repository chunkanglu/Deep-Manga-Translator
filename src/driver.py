from translation import Translation

def main():
    tr = Translation()
    img_path = r"C:\Users\Kang\Documents\Machine Learning\Manga_thing\manga_bubbles_test_1\images\10482900-Densetsu_no_ryuusou_14_1.jpg"
    out_path = r"res.png"

    output_img = tr.translate(img_path)

    output_img.show()
    output_img.save(out_path)

if __name__ == "__main__":
    main()
