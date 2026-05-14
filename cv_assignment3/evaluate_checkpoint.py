#!/usr/bin/env python3
"""
Load a checkpoint saved by cv_assignment3.main and report detailed test metrics.

Example:
  python -m cv_assignment3.evaluate_checkpoint --checkpoint ./checkpoints/best_lenet.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from cv_assignment3.data import CIFAR10_CLASSES, get_cifar10_loaders
from cv_assignment3.main import build_model, pick_device
from cv_assignment3.metrics import confusion_matrix_to_csv_rows, detailed_test_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved CIFAR-10 checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--out-json", type=str, default=None, help="Optional path to write full report JSON")
    p.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
        help="Limit test loader to this many batches (recommended for NASNet on CPU).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_name = ckpt["model"]
    img_size = int(ckpt["img_size"])
    pretrained = bool(ckpt.get("pretrained", False))
    nasnet_variant = ckpt.get("nasnet_variant") or "mobile"

    _, _, test_loader = get_cifar10_loaders(
        model_name=model_name,
        batch_size=args.batch_size,
        img_size=img_size,
        pretrained=pretrained,
        num_workers=args.num_workers,
        seed=args.seed,
        data_dir=args.data_dir,
    )

    model = build_model(model_name, pretrained=pretrained, nasnet_variant=nasnet_variant).to(device)
    model.load_state_dict(ckpt["model_state"])
    criterion = nn.CrossEntropyLoss()

    detail = detailed_test_report(
        model, test_loader, device, criterion=criterion, max_batches=args.max_test_batches
    )
    detail["model"] = model_name
    detail["checkpoint"] = str(ckpt_path.resolve())
    detail["epoch_saved"] = ckpt.get("epoch")
    detail["val_acc_at_save_pct"] = ckpt.get("val_acc")
    detail["class_names"] = list(CIFAR10_CLASSES)

    print(f"Model: {model_name} | checkpoint: {ckpt_path}")
    if "mean_loss" in detail:
        print(f"Mean test loss: {detail['mean_loss']:.4f}")
    print(f"Overall test accuracy: {detail['overall_accuracy_pct']:.2f}%")
    print("Per-class accuracy (%):")
    for name, acc, sup in zip(
        CIFAR10_CLASSES,
        detail["per_class_accuracy_pct"],
        detail["per_class_support"],
    ):
        print(f"  {name:12s}  {acc:6.2f}%  (n={sup})")

    out_json = args.out_json or str(ckpt_path.parent / f"eval_{model_name}_detailed.json")
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2)
    print(f"Wrote JSON report to {out_path.resolve()}")

    csv_path = out_path.with_suffix(".confusion.csv")
    rows = confusion_matrix_to_csv_rows(detail["confusion_matrix"], CIFAR10_CLASSES)
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote confusion matrix CSV to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
