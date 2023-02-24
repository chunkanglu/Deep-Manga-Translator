from translation import Translation

def main():
    tr = Translation()
    img_path = r"manga_bubbles_train_2\images\fcc05481-Dandadan_25_5.jpg"
    out_path = r"test_output/test.png"

    tr.translate(img_path, out_path)

if __name__ == "__main__":
    main()