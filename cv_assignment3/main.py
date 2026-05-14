#!/usr/bin/env python3
"""
Train and evaluate LeNet-5 or NASNet on CIFAR-10.

Examples:
  python -m cv_assignment3.main --model lenet --epochs 20 --batch-size 128
  python -m cv_assignment3.main --model nasnet --epochs 5 --batch-size 16 --img-size 224
  python -m cv_assignment3.main --model nasnet --pretrained --epochs 10 --batch-size 8 --img-size 331

CPU / smoke NASNet (caps part of each epoch — document in report; use full runs on GPU):

  python -m cv_assignment3.main --model nasnet --epochs 8 --batch-size 8 --img-size 96 \\
    --max-train-batches 120 --max-val-batches 40
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from cv_assignment3.data import CIFAR10_CLASSES, get_cifar10_loaders
from cv_assignment3.metrics import detailed_test_report
from cv_assignment3.models.lenet5 import LeNet5CIFAR
from cv_assignment3.models.nasnet import build_nasnet, nasnet_suggested_input_size


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item() * 100.0


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy(logits.detach(), labels)
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        total_acc += accuracy(logits, labels)
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def pick_device(prefer: str) -> torch.device:
    prefer = prefer.lower()
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


def build_model(model_name: str, pretrained: bool, nasnet_variant: str = "mobile") -> nn.Module:
    if model_name == "lenet":
        return LeNet5CIFAR(num_classes=10)
    if model_name == "nasnet":
        return build_nasnet(num_classes=10, pretrained=pretrained, variant=nasnet_variant)
    raise ValueError(f"Unknown model {model_name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIFAR-10: LeNet-5 or NASNet training")
    p.add_argument("--model", choices=["lenet", "nasnet"], required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="If unset, uses a sensible default per model.")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--img-size", type=int, default=None, help="Spatial size for NASNet path (ignored for LeNet).")
    p.add_argument("--pretrained", action="store_true", help="ImageNet weights for NASNet (download).")
    p.add_argument(
        "--nasnet-variant",
        choices=["mobile", "large"],
        default="mobile",
        help="NASNet-A variant. 'mobile' (~5M params, CPU-friendly) is default; 'large' (~85M) needs GPU.",
    )
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./checkpoints")
    p.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Stop each training epoch after this many mini-batches (e.g. CPU-friendly NASNet).",
    )
    p.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Stop each validation pass after this many mini-batches (must set with train cap for fair CPU demos).",
    )
    p.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
        help="Cap final test metrics to this many batches. Default: if --max-train-batches is set, uses min(80, full test loader length); else full test set.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    if args.model == "nasnet":
        img_size = args.img_size or nasnet_suggested_input_size(args.pretrained, args.nasnet_variant)
        default_lr = 3e-4 if args.pretrained else 1e-3
    else:
        img_size = 32
        default_lr = 1e-3

    lr = args.lr if args.lr is not None else default_lr

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        model_name=args.model,
        batch_size=args.batch_size,
        img_size=img_size,
        pretrained=args.pretrained,
        num_workers=args.num_workers,
        seed=args.seed,
        data_dir=args.data_dir,
    )

    max_test_batches = args.max_test_batches
    if max_test_batches is None and args.max_train_batches is not None:
        max_test_batches = min(80, len(test_loader))

    variant_note = f" [{args.nasnet_variant}]" if args.model == "nasnet" else ""
    print(f"Building model ({args.model}{variant_note})…", flush=True)
    model = build_model(args.model, pretrained=args.pretrained, nasnet_variant=args.nasnet_variant).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history: Dict[str, list] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = -1.0
    best_path = out_dir / f"best_{args.model}.pt"

    cap_note = ""
    if args.max_train_batches is not None:
        cap_note += f" | train≤{args.max_train_batches}b/epoch"
    if args.max_val_batches is not None:
        cap_note += f" | val≤{args.max_val_batches}b/epoch"
    print(f"Device: {device} | Model: {args.model} | img_size={img_size} | lr={lr}{cap_note}", flush=True)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_batches=args.max_train_batches
        )
        va_loss, va_acc = evaluate(
            model, val_loader, criterion, device, max_batches=args.max_val_batches
        )
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.2f}% | "
            f"val loss {va_loss:.4f} acc {va_acc:.2f}%",
            flush=True,
        )
        if va_acc > best_val:
            best_val = va_acc
            torch.save(
                {
                    "model": args.model,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": va_acc,
                    "img_size": img_size,
                    "pretrained": args.pretrained,
                    "nasnet_variant": args.nasnet_variant if args.model == "nasnet" else None,
                    "max_train_batches": args.max_train_batches,
                    "max_val_batches": args.max_val_batches,
                },
                best_path,
            )

    # Load best weights for official test evaluation (detailed metrics, single pass).
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # Apple MPS: (1) CrossEntropyLoss over long test loops can raise AcceleratorError.
    # (2) Moving an MPS-trained module with .cpu() and evaluating can yield degenerate
    # predictions (~10% acc, all-one-class). Build a fresh CPU model and load weights
    # with map_location="cpu" for a trustworthy test pass.
    if device.type == "mps":
        ckpt_cpu = torch.load(best_path, map_location="cpu", weights_only=False)
        eval_model = build_model(
            args.model, args.pretrained, nasnet_variant=args.nasnet_variant
        ).cpu()
        eval_model.load_state_dict(ckpt_cpu["model_state"])
        test_detail = detailed_test_report(
            eval_model,
            test_loader,
            torch.device("cpu"),
            criterion=criterion,
            max_batches=max_test_batches,
        )
        test_detail["test_eval_device"] = "cpu (fresh model; MPS-safe)"
    else:
        test_detail = detailed_test_report(
            model, test_loader, device, criterion=criterion, max_batches=max_test_batches
        )
    test_detail["class_names"] = list(CIFAR10_CLASSES)
    te_acc = test_detail["overall_accuracy_pct"]
    te_loss = test_detail["mean_loss"]
    elapsed = time.time() - t0
    print(f"Best val acc: {best_val:.2f}% | Test acc: {te_acc:.2f}% | Time: {elapsed/60:.1f} min", flush=True)

    n_params = sum(p.numel() for p in model.parameters())

    summary = {
        "model": args.model,
        "nasnet_variant": args.nasnet_variant if args.model == "nasnet" else None,
        "best_val_acc": best_val,
        "test_acc": te_acc,
        "test_loss": te_loss,
        "num_parameters": n_params,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": lr,
        "img_size": img_size,
        "pretrained": args.pretrained,
        "max_train_batches_per_epoch": args.max_train_batches,
        "max_val_batches_per_epoch": args.max_val_batches,
        "max_test_batches": max_test_batches,
        "seconds": elapsed,
    }
    with open(out_dir / f"summary_{args.model}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / f"history_{args.model}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / f"test_report_{args.model}.json", "w", encoding="utf-8") as f:
        json.dump(test_detail, f, indent=2)


if __name__ == "__main__":
    main()
