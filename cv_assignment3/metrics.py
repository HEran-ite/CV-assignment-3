"""Extra evaluation metrics for reports (per-class accuracy, confusion matrix)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


@torch.no_grad()
def detailed_test_report(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 10,
    criterion: nn.Module | None = None,
    max_batches: int | None = None,
) -> Dict[str, Any]:
    """
    Run the model on the loader and aggregate:
      - mean cross-entropy loss (if criterion is provided)
      - overall accuracy (%)
      - per-class accuracy (%) with support (count) per class
      - confusion matrix [true][pred] as nested lists (rows = true label)

    If max_batches is set, metrics are computed only on the first N batches
    (useful for NASNet on CPU); the report then includes partial_test=true.
    """
    model.eval()
    correct_per = torch.zeros(num_classes, device=device)
    total_per = torch.zeros(num_classes, device=device)
    confusion = torch.zeros(num_classes, num_classes, device=device, dtype=torch.long)
    loss_sum = 0.0
    n_samples = 0
    batches_done = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            loss_sum += criterion(logits, labels).item() * labels.size(0)
        n_samples += labels.size(0)
        pred = logits.argmax(dim=1)
        eq = pred == labels
        for c in range(num_classes):
            mask = labels == c
            total_per[c] += mask.sum()
            correct_per[c] += (eq & mask).sum()
        idx = labels.long() * num_classes + pred.long()
        confusion.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
        batches_done += 1
        if max_batches is not None and batches_done >= max_batches:
            break

    total = total_per.sum().clamp(min=1)
    overall = (correct_per.sum() / total).item() * 100.0
    per_class_acc: List[float] = (correct_per / total_per.clamp(min=1)).tolist()
    support: List[int] = total_per.long().tolist()

    out: Dict[str, Any] = {
        "overall_accuracy_pct": overall,
        "per_class_accuracy_pct": per_class_acc,
        "per_class_support": support,
        "confusion_matrix": confusion.cpu().tolist(),
        "partial_test": max_batches is not None,
        "test_batches_used": batches_done,
    }
    if criterion is not None and n_samples > 0:
        out["mean_loss"] = loss_sum / n_samples
    return out


def confusion_matrix_to_csv_rows(matrix: List[List[int]], class_names: Sequence[str]) -> List[str]:
    """Header row + one row per true class for pasting into a spreadsheet."""
    header = ["true \\ pred"] + list(class_names)
    lines = [",".join(header)]
    for i, row in enumerate(matrix):
        lines.append(",".join([class_names[i]] + [str(x) for x in row]))
    return lines
