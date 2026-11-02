# Results and limitations

## Historical Kaggle output

The rendered [version 3 Kaggle notebook](https://www.kaggle.com/code/arianabdollahi/image-captioning) was inspected on 2026-11-03. Its saved training cell displays these values:

| Displayed epoch | Training loss | Training reported perplexity | Validation loss | Validation reported perplexity |
| --- | --- | --- | --- | --- |
| 0 | 2.05 | 82.5 | 2.238 | 105.4 |
| 1 | 2.05 | 81.8 | 2.236 | 104.0 |

These are transcribed historical outputs, not a new run. The code returned by Kaggle's pull API contains source cells without the displayed outputs. The notebook also displays autoregressive image-caption examples, but no verified BLEU, METEOR, CIDEr, or SPICE score is available.

The notebook's captions are serialized Python lists, and its perplexity metric does not exclude padding in the same way as its cross-entropy. The historical loss and reported perplexity are therefore not internally comparable as `perplexity = exp(loss)`. They must not be used as a benchmark for the corrected package.

## Refactored package

The package has been checked with unit tests and a one-epoch synthetic CPU run using generated solid-color images and repeated short captions. The run confirms that the training, checkpoint, evaluation, and prediction paths execute. Its generated captions and metrics do not measure real-image captioning ability and are intentionally not presented as model performance results.

No full Flickr30k training or GPU validation was performed. No pretrained captioning checkpoint is provided. A meaningful result requires training on the real dataset, retaining a held-out image split, evaluating freely generated captions against all references, and reporting the dataset split, environment, and metric implementation.
