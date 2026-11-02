"""Teacher-forced training and evaluation with padding-aware token metrics."""

import math

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .vocabulary import PAD_IDX


def run_epoch(model, loader, device, optimizer=None, gradient_clip=0.5, progress=True):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    batches = tqdm(loader, desc="Train" if training else "Evaluate", disable=not progress)
    with torch.set_grad_enabled(training):
        for images, captions in batches:
            images = images.to(device)
            captions = captions.to(device)
            inputs, targets = captions[:, :-1], captions[:, 1:]
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images, inputs)
            token_count = int(targets.ne(PAD_IDX).sum().item())
            if not token_count:
                raise ValueError("Batch has no non-padding target tokens")
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=PAD_IDX,
                reduction="sum",
            )
            loss = loss_sum / token_count
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; check data and learning rate")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()
            total_loss += loss_sum.detach().item()
            total_tokens += token_count
            batches.set_postfix(loss=f"{total_loss / total_tokens:.4f}")
    if not total_tokens:
        raise ValueError("Data loader produced no target tokens")
    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss), "tokens": total_tokens}
