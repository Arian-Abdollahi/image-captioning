"""Flickr30k annotations, image-level splits, augmentation, and padded batches."""

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .vocabulary import PAD_IDX

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    captions: tuple[str, ...]


def read_annotations(path):
    """Read the notebook's pipe-separated image_name | index | caption format."""
    grouped = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        header = next(reader, None)
        if header is None or len(header) < 3:
            raise ValueError("Expected a header and three pipe-separated annotation columns")
        for line_number, row in enumerate(reader, start=2):
            if not row or not any(part.strip() for part in row):
                continue
            if len(row) < 3:
                raise ValueError(f"Invalid annotation row {line_number}: expected three columns")
            image_id = row[0].strip()
            caption = " ".join("|".join(row[2:]).strip().lower().split())
            if not image_id or not caption or caption == "nan":
                continue
            if Path(image_id).name != image_id or "\\" in image_id:
                raise ValueError(f"Expected an image filename on row {line_number}")
            grouped.setdefault(image_id, []).append(caption)
    if not grouped:
        raise ValueError("No valid image captions found")
    return [ImageRecord(key, tuple(grouped[key])) for key in sorted(grouped)]


def split_records(records, validation_fraction=0.1, test_fraction=0.1, seed=42):
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Split fractions must be in (0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to < 1")
    records = sorted(records, key=lambda record: record.image_id)
    if len({record.image_id for record in records}) != len(records):
        raise ValueError("Split input must contain unique image IDs")
    random.Random(seed).shuffle(records)
    num_validation = max(1, int(len(records) * validation_fraction))
    num_test = max(1, int(len(records) * test_fraction))
    if len(records) - num_validation - num_test < 1:
        raise ValueError("At least three unique images are required for train/validation/test")
    return {
        "train": records[num_validation + num_test :],
        "validation": records[:num_validation],
        "test": records[num_validation : num_validation + num_test],
    }


def make_transforms(resize=356, crop_size=299, training=False):
    operations = [transforms.Resize((resize, resize))]
    if training:
        operations.extend(
            [
                transforms.RandomCrop((crop_size, crop_size)),
                transforms.RandomHorizontalFlip(),
            ]
        )
    else:
        operations.append(transforms.CenterCrop((crop_size, crop_size)))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


class FlickrDataset(Dataset):
    """Use each human caption as its own target, retaining image-level split boundaries."""

    def __init__(self, records, image_dir, vocabulary, transform, max_length=100):
        self.image_dir = Path(image_dir)
        self.vocabulary = vocabulary
        self.transform = transform
        self.max_length = max_length
        self.samples = [
            (record.image_id, caption) for record in records for caption in record.captions
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_id, caption = self.samples[index]
        path = self.image_dir / image_id
        # A missing or corrupt image is a real data error; do not train on a blank substitute.
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        tokens = self.vocabulary.encode(caption, self.max_length)
        return tensor, torch.tensor(tokens, dtype=torch.long)


def collate_captions(batch):
    images, captions = zip(*batch)
    return torch.stack(images), pad_sequence(captions, batch_first=True, padding_value=PAD_IDX)


def _seed_worker(_worker_id):
    import numpy as np

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(records, config, vocabulary, *, training=False, device=None):
    dataset = FlickrDataset(
        records,
        config.image_dir,
        vocabulary,
        make_transforms(config.resize, config.crop_size, training=training),
        config.max_length,
    )
    if training and len(dataset) < config.batch_size:
        raise ValueError("Training captions are fewer than batch_size; lower batch_size to >= 2")
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=config.num_workers,
        pin_memory=device is not None and device.type == "cuda",
        collate_fn=collate_captions,
        drop_last=training,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
