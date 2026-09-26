import torch
from torch import Tensor, nn
from transformer.model_cfg import *
import torch.nn.functional as F


def prediction_metrics_from_logits(logits: Tensor, targets: Tensor, is_training: bool) -> dict[str, int]:

    """Count predicted UP, DOWN, and FLAT classes in one batch."""
    if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"logits must have shape [batch, {len(CLASS_NAMES)}]")
    predicted_classes = logits.detach().argmax(dim=1)
    counts = torch.bincount(
        predicted_classes, minlength=len(CLASS_NAMES)
    ).cpu()
    metrics = {
        class_name: int(counts[index])
        for index, class_name in enumerate(CLASS_NAMES)
    }
    detached_targets = targets.detach()
    target_counts = torch.bincount(
        detached_targets, minlength=len(CLASS_NAMES)
    ).cpu()

    metrics.update(
        {
            f"{class_name}_TARGET": int(target_counts[index])
            for index, class_name in enumerate(CLASS_NAMES)
        }
    )

    softmax_output = F.softmax(logits, dim=1)
    softmax_output = softmax_output.cpu().tolist()
    for index, vals in enumerate(softmax_output):
        softmax_output[index] = [round(val, 3) for val in vals]

    if not is_training:
        print(f"--- softmax_output: {softmax_output}\n")
        print(f"--- target_output: {detached_targets.tolist()}\n")

    return metrics
