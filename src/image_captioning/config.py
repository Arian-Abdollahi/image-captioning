"""Experiment configuration; paths are relative to the working directory."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    captions_file: str = "data/flickr30k/results.csv"
    image_dir: str = "data/flickr30k/images"
    output_dir: str = "artifacts"
    seed: int = 42
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    resize: int = 356
    crop_size: int = 299
    max_length: int = 100
    min_frequency: int = 1
    embed_size: int = 300
    num_heads: int = 10
    num_layers: int = 3
    dropout: float = 0.1
    pretrained_encoder: bool = True
    use_word2vec: bool = True
    word2vec_window: int = 20
    word2vec_min_count: int = 5
    word2vec_epochs: int = 5
    word2vec_workers: int = 1
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 2
    learning_rate: float = 0.005
    weight_decay: float = 1e-4
    momentum: float = 0.9
    gradient_clip: float = 0.5

    def validate(self):
        for name in (
            "seed",
            "resize",
            "crop_size",
            "max_length",
            "min_frequency",
            "embed_size",
            "num_heads",
            "num_layers",
            "word2vec_window",
            "word2vec_min_count",
            "word2vec_epochs",
            "word2vec_workers",
            "batch_size",
            "num_workers",
            "epochs",
        ):
            value = getattr(self, name)
            minimum = 0 if name in {"seed", "num_workers"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2 for the encoder projection BatchNorm")
        if self.max_length < 3:
            raise ValueError("max_length must allow <SOS>, a word, and <EOS>")
        if self.embed_size % self.num_heads:
            raise ValueError("embed_size must be divisible by num_heads")
        if self.resize < self.crop_size:
            raise ValueError("resize must be >= crop_size")
        if not 0 < self.validation_fraction < 1 or not 0 < self.test_fraction < 1:
            raise ValueError("validation_fraction and test_fraction must be in (0, 1)")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be < 1")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate/gradient_clip must be positive; weight_decay >= 0")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if type(self.pretrained_encoder) is not bool or type(self.use_word2vec) is not bool:
            raise ValueError("pretrained_encoder and use_word2vec must be JSON booleans")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**values).validate()

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
