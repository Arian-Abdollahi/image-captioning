"""Train a captioning model: python -m image_captioning.train --help."""

import argparse
import platform
from pathlib import Path

import torch
import torchvision

from .checkpoint import annotation_digest, save_checkpoint
from .config import Config
from .data import make_loader, read_annotations, split_records
from .embeddings import build_embedding_weights
from .engine import run_epoch
from .model import ImageCaptioningModel
from .utils import seed_everything, select_device, write_json
from .vocabulary import Vocabulary


def train(config, device="auto"):
    config.validate()
    device = select_device(device)
    seed_everything(config.seed)
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; choose a new output_dir")

    records = read_annotations(config.captions_file)
    splits = split_records(
        records,
        config.validation_fraction,
        config.test_fraction,
        config.seed,
    )
    train_captions = [caption for record in splits["train"] for caption in record.captions]
    vocabulary = Vocabulary.build(train_captions, config.min_frequency)
    if len(vocabulary) <= 4:
        raise ValueError("Training captions produced no vocabulary; check min_frequency")
    train_loader = make_loader(splits["train"], config, vocabulary, training=True, device=device)
    validation_loader = make_loader(splits["validation"], config, vocabulary, device=device)
    embedding_weights = (
        build_embedding_weights(train_captions, vocabulary, config) if config.use_word2vec else None
    )
    model = ImageCaptioningModel(config, len(vocabulary), embedding_weights).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    split_ids = {name: [record.image_id for record in items] for name, items in splits.items()}
    digest = annotation_digest(config.captions_file)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config.to_dict())
    write_json(output / "vocabulary.json", vocabulary.word_to_index)
    write_json(
        output / "splits.json",
        {
            "seed": config.seed,
            "annotation_sha256": digest,
            "image_ids": split_ids,
            "counts": {
                name: {"images": len(items), "captions": sum(len(r.captions) for r in items)}
                for name, items in splits.items()
            },
        },
    )
    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "device": str(device),
        },
    )
    best_loss = float("inf")
    history = []
    print(
        f"Device: {device}; vocabulary: {len(vocabulary)}; splits: "
        f"{ {name: len(items) for name, items in splits.items()} }"
    )
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, config.gradient_clip)
        validation_metrics = run_epoch(model, validation_loader, device)
        metrics = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(metrics)
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            save_checkpoint(
                output / "best.pt",
                model,
                optimizer,
                config,
                vocabulary,
                split_ids,
                digest,
                epoch,
                metrics,
            )
        save_checkpoint(
            output / "last.pt",
            model,
            optimizer,
            config,
            vocabulary,
            split_ids,
            digest,
            epoch,
            metrics,
        )
        write_json(output / "history.json", history)
        print(
            f"Epoch {epoch}: train loss={train_metrics['loss']:.4f}; "
            f"validation loss={validation_metrics['loss']:.4f}; "
            f"validation perplexity={validation_metrics['perplexity']:.2f}"
        )
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--captions-file")
    parser.add_argument("--image-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument(
        "--no-pretrained", action="store_true", help="Do not download ImageNet weights"
    )
    parser.add_argument(
        "--no-word2vec", action="store_true", help="Use random trainable embeddings"
    )
    args = parser.parse_args(argv)
    config = Config.load(args.config)
    for field in (
        "captions_file",
        "image_dir",
        "output_dir",
        "epochs",
        "batch_size",
        "num_workers",
    ):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
    if args.no_pretrained:
        config.pretrained_encoder = False
    if args.no_word2vec:
        config.use_word2vec = False
    train(config, args.device)


if __name__ == "__main__":
    main()
