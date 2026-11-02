# Verification

Verified on **2026-11-03** using Python **3.12.14** and CPU PyTorch. Top-level versions are recorded in [requirements-tested.txt](../requirements-tested.txt).

| Check | Result |
| --- | --- |
| Editable package installation | Passed |
| Dependency consistency (`pip check`) | Passed |
| Ruff lint on `src/`, `tests/`, and `scripts/` | Passed |
| Original notebook | All 52 cells parsed as notebook JSON; code cells parsed as Python; source SHA-256 matches the retrieved source |
| Workflow notebook | All 12 cells parsed as notebook JSON; code cells parsed as Python |
| Unit tests | 10 passed |
| Synthetic CLI workflow | One CPU training epoch, best-checkpoint reload, test evaluation, and autoregressive prediction passed |
| Annotation mismatch guard | Evaluation rejected a changed annotation file |

## Unit-test coverage

`python -m unittest discover -s tests -v` checks that caption references stay separate, image splits are disjoint and repeatable, unknown words map correctly, truncation preserves `<EOS>`, batches are padded with the expected layout, deterministic transforms repeat, malformed annotations and missing images fail clearly, Word2Vec initialization repeats and has a zero padding row, the decoder cannot attend to future tokens, padding does not change token metrics, the projection learns while the CNN remains frozen, checkpoints preserve predictions after reload, and incompatible attention dimensions fail early.

## Synthetic integration run

`python scripts/smoke_test.py` creates eight solid-color images with two captions each in a temporary directory. It trains for one epoch with a small Transformer, actual untrained ResNet-50, Word2Vec initialization, and CPU execution; then invokes the evaluation and prediction CLIs in separate Python processes. It makes no pretrained-weight or dataset download. Temporary checkpoints and fixtures are deleted afterward.

The test confirms execution and serialization, not caption quality. Neither a pretrained encoder download nor a full Flickr30k experiment was run during verification. CUDA, multiworker data loading, real-data convergence, generation-level metrics, and the historical notebook's original runtime environment remain unverified. The original notebook is retained for inspection and was not executed end-to-end.
