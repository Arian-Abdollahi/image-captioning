# From Kaggle notebook to Python project

Source: [Image_Captioning, Arian Abdollahi, version 3](https://www.kaggle.com/code/arianabdollahi/image-captioning). The API returned 52 source cells without saved outputs. That source is preserved byte-for-byte in `notebooks/image_captioning_original.ipynb` (SHA-256 `2541f47bdba017519f00e38828c25240d14105c2801fe9161923ccc786711d17`). The rendered Kaggle page separately displays historical outputs.

The package keeps the central architecture: frozen ResNet-50, linear/BatchNorm image projection, Word2Vec-initialized trainable embeddings, sinusoidal positional encoding, and a Transformer decoder. The configured defaults use 300-dimensional embeddings and **10** attention heads, matching the notebook's actual model construction. Its unused `num_heads = 8` variable would not divide 300 evenly.

## Changes that affect behavior

| Notebook behavior | Package behavior | Reason |
| --- | --- | --- |
| Groups caption lists, then calls `astype(str)` | Keeps captions as individual strings within image groups | Avoid learning Python list punctuation and concatenated reference captions |
| Supplies raw strings to Word2Vec | Supplies lists of word tokens | Train word embeddings rather than iterating string characters |
| Random image-level split with no seed | Seeded, saved image-level split | Repeat the same split and preserve all references for each image |
| Vocabulary ties depend on encounter order | Frequency sort with lexical tie-breaking | Stable vocabulary IDs |
| Loads an English pipeline and runs its components | Uses `spacy.blank('en').tokenizer` | Tokenization does not require tagging, parsing, NER, or a model download |
| Random crops/flips for validation and test; 224-pixel external-image resize | Random transforms only for training; shared deterministic resize/center crop for evaluation and prediction | Consistent evaluation and inference preprocessing |
| `ColorJitter()` with all defaults | Omits that no-op transform | The original call has no active jitter settings |
| Missing images become blank placeholders | Image errors propagate | Make broken data visible |
| Naive sequence truncation can remove `<EOS>` | Reserves slots for `<SOS>` and `<EOS>` | Retain a stopping target |
| Decoder has only a causal mask | Adds a caption padding mask too | Exclude padding keys in self-attention |
| Freezes CNN parameters but training mode updates backbone BatchNorm statistics | Keeps the frozen backbone in evaluation mode | Freeze running statistics as well as gradients |
| May produce singleton training batches | Drops the final incomplete training batch | Avoid BatchNorm failure with one sample |
| Cross-entropy ignores padding; perplexity does not specify `ignore_index` | Both derive from the same non-padding token loss | Make loss and perplexity consistent |
| Averages batch losses by image count | Aggregates loss sums over target-token counts | Correct corpus-level token weighting |
| Saves the entire model object; global histories and paths | Saves state dictionaries with config, vocabulary, image IDs, annotation hash, and metrics | Portable reload without notebook global state |
| Applies temperature to greedy logits | Uses plain greedy decoding | Positive scalar temperature does not change argmax; no sampling is implemented |
| Teacher-forced argmax previews mixed with generation utilities | Explicitly distinguishes teacher-forced evaluation and autoregressive prediction | Avoid treating reference-conditioned previews as free generation |

The original and refactored implementations therefore need separate experimental results. Existing notebook weights are not directly supported by the package's loader; retrain with this package to produce compatible checkpoints and vocabulary IDs.

## Reproducibility

`configs/default.json` is the experiment configuration. The run saves it together with image IDs, an annotation SHA-256 digest, vocabulary, history, and basic Python/PyTorch/torchvision versions. Word2Vec uses a stable initialization hash and one worker by default. Python, NumPy, PyTorch, and data-loader seeds are set; cuDNN benchmarking is disabled.

These controls do not promise bitwise equivalence across devices, operating systems, or library versions. `requirements-tested.txt` records top-level versions used for local verification, not a complete transitive or CUDA lockfile. The GPU path has not been tested.

Training output directories must be empty. There is no resume CLI or optimizer-resume workflow, although optimizer state is retained in checkpoints. The last checkpoint supports model inspection; inference normally uses the best validation checkpoint.

Checkpoints use `torch.load(..., weights_only=True)` and contain tensors and plain metadata. Loading a checkpoint does not initiate an ImageNet download. Evaluation verifies the annotation file's exact bytes before resolving the saved held-out IDs; a semantically equivalent but edited CSV will intentionally fail this check.

## Metrics and remaining scope

Teacher-forced evaluation shifts captions by one token, predicts the next word, masks `<PAD>` targets, and computes perplexity as `exp(mean token cross-entropy)`. `<EOS>` is included as a target. Inference encodes an image once, starts from `<SOS>`, feeds back generated tokens, suppresses `<PAD>`/`<SOS>` as output words, and stops at `<EOS>` or the configured limit.

There is no beam search, sampling decoder, spatial-attention encoder, generation-level caption metric suite, or trained checkpoint in this repository. Full Flickr30k training and generation evaluation remain to be run.

## Implementation references

- [PyTorch ResNet-50 weights API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html): explicit `IMAGENET1K_V1` preserves the original `pretrained=True` choice.
- [PyTorch TransformerDecoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html): causal and target padding masks.
- [PyTorch model serialization](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html): state-dict save/load pattern.
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html): tokenized sentence input.
