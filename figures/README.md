# Result figure provenance

`kaggle_caption_dog.png` is an unmodified PNG from the saved output of [Arian Abdollahi's Image_Captioning notebook on Kaggle](https://www.kaggle.com/code/arianabdollahi/image-captioning), version 3. It was retrieved on 2026-11-03.

- **Notebook cell:** `In [154]`, fourth displayed example, generated from `test_dataset[303]`.
- **Rendered output file:** `__results___files/__results___49_6.png`.
- **Dimensions:** 389 × 389 pixels.
- **Method:** the notebook's `generate_caption` function performs autoregressive greedy decoding without a reference caption.
- **Generated caption text:** `a black dog is swimming in the water with a yellow ball in its mouth .`

The README removes surrounding list-format characters from the saved text and normalizes the space before the final period. The image itself is unchanged. The other examples were reviewed visually; this example was selected because the subject, action, and ball description closely match the displayed image. This is a qualitative illustration from the original notebook, not a new evaluation result for the refactored package.

The notebook uses the [Flickr Image dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset). The image retains its original dataset terms; the repository's Apache code license does not relicense it.
