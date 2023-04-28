from translation import Translation

def main():
    tr = Translation()
    img_path = r""
    out_path = r""

    output_img = tr.translate(img_path)

    output_img.show()
    output_img.save(output_path)

if __name__ == "__main__":
    main()