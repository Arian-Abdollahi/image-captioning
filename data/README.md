# Flickr30k data setup

The original notebook reads [hsankesara/flickr-image-dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset). Obtain the data from its provider and follow the dataset's access and usage terms. Full training datasets, annotation files, and private test photos are not included. One saved notebook result figure is reproduced in the README; see [figure provenance](../figures/README.md).

The package expects a header followed by **pipe-separated** rows, one caption per row:

```text
image_name|comment_number|comment
1000092795.jpg|0|Two people are outdoors.
1000092795.jpg|1|A pair of people stand together.
```

These two captions are illustrative examples of the file format. The annotation reader uses the first column as the image filename and the third as caption text, ignoring the caption index. Leading/trailing spaces are removed, text is lowercased, whitespace is normalized, and missing captions are skipped. Malformed rows raise an error with the line number.

Default local locations:

| File | Location |
| --- | --- |
| Annotation table | `data/flickr30k/results.csv` |
| Image example | `data/flickr30k/images/1000092795.jpg` |

The Kaggle notebook's mounted locations are:

```text
/kaggle/input/flickr-image-dataset/flickr30k_images/results.csv
/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images/
```

You do not have to move or rename the images: set `captions_file` and `image_dir` in the config, or use the CLI overrides. Paths are relative to the current working directory unless absolute. Image filenames in the CSV must be simple filenames rather than absolute paths.

The training script groups rows by filename, creates reproducible image-level splits, and then expands captions within each split into separate targets. With the default fractions, each split has at least one image and at least three unique images are required overall. Use enough training captions for at least one batch, or lower `batch_size` to a value of at least 2.

The training loader drops its final incomplete batch because the image projection uses BatchNorm. Validation and test keep every example. Missing or corrupt images raise an error instead of being silently replaced with a blank image.

The private test-photo input referenced by the notebook is not needed for this package. Use a local photo with `image_captioning.predict` after training.
