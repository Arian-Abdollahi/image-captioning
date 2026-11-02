"""Optional Matplotlib utilities for notebook use."""

import json
from pathlib import Path

from .data import IMAGENET_MEAN, IMAGENET_STD


def show_image(tensor, caption=None):
    import matplotlib.pyplot as plt
    import torch

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = (tensor.detach().cpu() * std + mean).clamp(0, 1)
    figure, axis = plt.subplots()
    axis.imshow(image.permute(1, 2, 0).numpy())
    axis.axis("off")
    if caption:
        axis.set_title(caption, wrap=True)
    return figure


def plot_history(path):
    import matplotlib.pyplot as plt

    history = json.loads(Path(path).read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [item["epoch"] for item in history]
    for axis, metric in zip(axes, ("loss", "perplexity")):
        for split in ("train", "validation"):
            axis.plot(epochs, [item[split][metric] for item in history], label=split)
        axis.set(xlabel="Epoch", ylabel=metric.capitalize())
        axis.grid(True)
        axis.legend()
    figure.tight_layout()
    return figure
