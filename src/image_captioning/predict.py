"""Generate an image caption without access to a ground-truth caption."""

import argparse

import torch
from PIL import Image

from .checkpoint import load_model
from .data import make_transforms
from .utils import select_device
from .vocabulary import EOS_IDX, PAD_IDX, SOS_IDX


@torch.inference_mode()
def generate_caption(model, image, vocabulary, max_tokens=20, decoder_max_length=100):
    if max_tokens < 1 or max_tokens > decoder_max_length:
        raise ValueError(f"max_tokens must be between 1 and {decoder_max_length}")
    model.eval()
    device = next(model.parameters()).device
    features = model.encoder(image.unsqueeze(0).to(device))
    generated = [SOS_IDX]
    for _ in range(max_tokens):
        inputs = torch.tensor([generated], dtype=torch.long, device=device)
        scores = model.decoder(features, inputs)[:, -1, :].clone()
        # Structural tokens are never output words.
        scores[:, [PAD_IDX, SOS_IDX]] = float("-inf")
        next_token = int(scores.argmax(dim=-1).item())
        generated.append(next_token)
        if next_token == EOS_IDX:
            break
    return vocabulary.decode(generated)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    device = select_device(args.device)
    model, vocabulary, config, _ = load_model(args.checkpoint, device)
    with Image.open(args.image) as image:
        tensor = make_transforms(config.resize, config.crop_size)(image.convert("RGB"))
    print(generate_caption(model, tensor, vocabulary, args.max_tokens, config.max_length))


if __name__ == "__main__":
    main()
