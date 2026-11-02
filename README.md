# Image Captioning

Generate natural-language descriptions of images using a **ResNet-50 encoder** and a **Transformer decoder**, trained on **Flickr30k** captions.

This project reorganizes [Arian Abdollahi's Kaggle notebook](https://www.kaggle.com/code/arianabdollahi/image-captioning) into a Python package with separate data, model, training, evaluation, and inference modules. The original notebook source is preserved in [notebooks/](notebooks/).

## Model

| Component | Default configuration |
| --- | --- |
| Image encoder | Frozen ImageNet-pretrained ResNet-50, global average pooling |
| Image projection | Linear layer and BatchNorm, 2048 → 300 dimensions |
| Word embeddings | Trainable 300-dimensional embeddings initialized with Word2Vec |
| Caption decoder | 3 Transformer decoder layers, 10 attention heads, sinusoidal positions |
| Training | Teacher forcing, padding-masked cross-entropy, SGD with momentum |
| Prediction | Autoregressive greedy decoding from `<SOS>` until `<EOS>` or a length limit |

The decoder cross-attends to one pooled image vector. The implementation does not use spatial image regions or a vision-language foundation model.

## Repository layout

| Path | Purpose |
| --- | --- |
| [src/image_captioning/data.py](src/image_captioning/data.py) | Flickr30k CSV parsing, image splits, datasets, transforms, padded batches |
| [src/image_captioning/vocabulary.py](src/image_captioning/vocabulary.py) | Word tokenization, vocabulary, token encoding and decoding |
| [src/image_captioning/embeddings.py](src/image_captioning/embeddings.py) | Word2Vec initialization from training captions |
| [src/image_captioning/model.py](src/image_captioning/model.py) | ResNet-50 encoder, positional encoding, Transformer decoder |
| [src/image_captioning/engine.py](src/image_captioning/engine.py) | Training and teacher-forced evaluation loops |
| [src/image_captioning/checkpoint.py](src/image_captioning/checkpoint.py) | Model state, vocabulary, configuration, and split metadata |
| [src/image_captioning/train.py](src/image_captioning/train.py) | Training command |
| [src/image_captioning/evaluate.py](src/image_captioning/evaluate.py) | Held-out loss and perplexity command |
| [src/image_captioning/predict.py](src/image_captioning/predict.py) | Caption generation for a new image |
| [src/image_captioning/visualization.py](src/image_captioning/visualization.py) | Optional image display and training plots |
| [configs/default.json](configs/default.json) | Dataset paths and experiment settings |
| [notebooks/01_workflow.ipynb](notebooks/01_workflow.ipynb) | Guided workflow using the package |
| [notebooks/image_captioning_original.ipynb](notebooks/image_captioning_original.ipynb) | Unmodified Kaggle notebook source, version 3 |
| [docs/REFACTOR.md](docs/REFACTOR.md) | Changes from the notebook and compatibility notes |
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | Checks performed and their limits |

## Installation

Python **3.12** was used for verification. Create a virtual environment from the repository root:

```bash
git clone https://github.com/Arian-Abdollahi/image-captioning.git
cd image-captioning
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CPU environment used for verification:
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tested.txt
```

On Windows, activate with `.venv\Scripts\activate`. For GPU training, install a matching CUDA build of PyTorch and torchvision using the [official selector](https://pytorch.org/get-started/locally/), then run `python -m pip install -r requirements.txt`. A GPU is useful for full-dataset training; CPU mode supports inspection and small runs.

Tokenization uses spaCy's blank English tokenizer, so `en_core_web_sm` does not need to be downloaded. The first default training run downloads the ImageNet ResNet-50 weights. Dataset files must be obtained separately.

## Prepare Flickr30k

The notebook uses the [Flickr Image dataset on Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset). Follow [data/README.md](data/README.md) for the annotation format and file locations.

Edit `captions_file` and `image_dir` in `configs/default.json`, or supply them on the command line. The defaults expect `data/flickr30k/results.csv` and JPEGs under `data/flickr30k/images/`.

Images are split approximately **80% training / 10% validation / 10% test** with seed 42. All captions for an image stay in that image's split. Each caption becomes a separate training target; vocabulary and Word2Vec fitting use only the training split.

## Train

```bash
python -m image_captioning.train --config configs/default.json

# Example using the paths from the Kaggle notebook:
python -m image_captioning.train \
  --captions-file /kaggle/input/flickr-image-dataset/flickr30k_images/results.csv \
  --image-dir /kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images \
  --output-dir /kaggle/working/captioning-run
```

The default is 2 epochs, batch size 32, learning rate 0.005, momentum 0.9, and weight decay 0.0001. Change these in the JSON config. Use `--num-workers 0` if your notebook environment cannot spawn workers. Choose a new output directory for each run; existing experiment files are protected from overwriting.

Training writes `best.pt`, `last.pt`, `history.json`, `vocabulary.json`, `config.json`, `splits.json`, and `environment.json` to `artifacts/` by default. Checkpoints include the vocabulary and exact image split, and the best checkpoint is selected by validation loss. No dataset or weights are committed to Git.

## Evaluate

```bash
python -m image_captioning.evaluate \
  --checkpoint artifacts/best.pt \
  --split test \
  --output artifacts/test_metrics.json
```

This reports token-weighted cross-entropy and perplexity under **teacher forcing**, excluding `<PAD>` targets. It uses the saved held-out image IDs and checks that the annotation file matches training. If files moved, supply `--captions-file` and `--image-dir` with their new locations.

These metrics measure next-token prediction with ground-truth prefixes. They do not measure the quality of freely generated captions. BLEU, METEOR, CIDEr, and SPICE are not implemented here.

## Generate a caption

```bash
python -m image_captioning.predict \
  --checkpoint artifacts/best.pt \
  --image path/to/photo.jpg \
  --max-tokens 20
```

Prediction requires a checkpoint trained with this package. It needs only the image and checkpoint, uses the training configuration's deterministic evaluation transform, and does not download encoder weights. `caption-train`, `caption-evaluate`, and `caption-predict` are equivalent installed commands.

## Results and verification

The Kaggle page displays a saved validation loss of **2.236** and a reported perplexity of **104.0**. These are historical notebook outputs. Caption preprocessing, splitting, and metric handling have changed in this package, so those values are **not results of the refactored implementation**. See [reports/results.md](reports/results.md).

The refactor passed 10 semantic tests and a small synthetic CPU train/evaluate/predict run. Full Flickr30k training, GPU execution, and generation quality have not been verified. Trained weights are not included.

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

For the guided notebook and optional plots, install `python -m pip install '.[notebooks]'` and run `jupyter lab`. Read [docs/REFACTOR.md](docs/REFACTOR.md) before comparing with the original experiment.

## License and credits

The Kaggle notebook is displayed under Apache License 2.0; this adaptation retains that license. See [LICENSE](LICENSE), [NOTICE](NOTICE), and [CREDITS.md](CREDITS.md). Dataset images, captions, and pretrained weights have their own terms and are not distributed with this repository.
