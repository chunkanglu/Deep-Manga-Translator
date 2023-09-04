# Deep Manga Translator


## Future Plans
- [ ] Improve/Train text segmentation, bubble segmentation models (eg. with color, contrast augmentations since light grey text is usually missed)
- [ ] Train own inpainting model
- [ ] Fix issue where text seg model connected components fails on large text where characters are very separated
- [ ] Fix issue for bubble seg model where same bubble can be predicted for multiple times
- [ ] Update Streamlit web app
- [ ] Add CLI interface
- [ ] Add Docker interface
- [ ] Separate text segmentation between sfx and actual text
- [ ] Refactor and clean code
- [ ] Decide to keep or remove Dectectron2 bubble segmentation model