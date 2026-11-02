"""Evaluate held-out captions using the checkpoint's saved image split."""

import argparse
import json

from .checkpoint import annotation_digest, load_model
from .data import make_loader, read_annotations
from .engine import run_epoch
from .utils import select_device, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--captions-file", help="Override the checkpoint's annotation path")
    parser.add_argument("--image-dir", help="Override the checkpoint's image path")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", help="Optional path for JSON metrics")
    args = parser.parse_args(argv)
    device = select_device(args.device)
    model, vocabulary, config, payload = load_model(args.checkpoint, device)
    if args.captions_file:
        config.captions_file = args.captions_file
    if args.image_dir:
        config.image_dir = args.image_dir
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    config.validate()
    if annotation_digest(config.captions_file) != payload["annotation_sha256"]:
        raise ValueError("Annotation file differs from training; supply the original CSV")
    by_id = {record.image_id: record for record in read_annotations(config.captions_file)}
    records = [by_id[image_id] for image_id in payload["split_ids"][args.split]]
    metrics = run_epoch(model, make_loader(records, config, vocabulary, device=device), device)
    result = {"split": args.split, "checkpoint_epoch": payload["epoch"], **metrics}
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.output:
        write_json(args.output, result)


if __name__ == "__main__":
    main()
