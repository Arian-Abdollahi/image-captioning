"""Portable state-dict checkpoints with the exact vocabulary and experiment metadata."""

import hashlib
from pathlib import Path

import torch

from .config import Config
from .model import ImageCaptioningModel
from .vocabulary import Vocabulary


def annotation_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(path, model, optimizer, config, vocabulary, split_ids, digest, epoch, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
            "vocabulary": vocabulary.word_to_index,
            "split_ids": split_ids,
            "annotation_sha256": digest,
            "epoch": epoch,
            "metrics": metrics,
        },
        temporary,
    )
    temporary.replace(path)


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format; use checkpoints from this package")
    config = Config.from_dict(payload["config"])
    vocabulary = Vocabulary(payload["vocabulary"])
    # Every encoder parameter is in the checkpoint; never download ImageNet weights here.
    model = ImageCaptioningModel(config, len(vocabulary), pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, vocabulary, config, payload
