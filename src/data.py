"""CIFAR-10 data pipelines: different spatial sizes / normalization for LeNet vs NASNet."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Literal, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.utils import extract_archive

# Standard CIFAR-10 normalization (used when training both models from scratch on CIFAR).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# CIFAR-10 label order matches torchvision.datasets.CIFAR10 default.
CIFAR10_CLASSES: Tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# timm / ImageNet-style recipe for NASNet-A Large pretrained weights (see timm cfg).
IMAGENET_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STD = (0.5, 0.5, 0.5)

_CIFAR_TGZ = "cifar-10-python.tar.gz"
# Same as torchvision.datasets.CIFAR10.tgz_md5 — catches truncated downloads.
_CIFAR_TGZ_MD5 = "c58f30108f718f92721af3b95e74349a"


def _try_extract_cifar10_tarball(data_dir: str) -> bool:
    """
    If the official archive is already on disk but not extracted, extract it locally.
    Avoids calling torchvision's download path (which always hits the network first).

    Returns True only if extraction was attempted and succeeded. Corrupt/incomplete
    archives (wrong MD5) are left alone so the caller can fall back to download=True.
    """
    from torchvision.datasets.utils import check_integrity

    root = Path(data_dir).expanduser()
    archive = root / _CIFAR_TGZ
    batch_dir = root / "cifar-10-batches-py"
    if batch_dir.is_dir():
        return False
    if not archive.is_file():
        return False
    if not check_integrity(str(archive), _CIFAR_TGZ_MD5):
        return False
    try:
        extract_archive(str(archive), str(root), remove_finished=False)
    except (EOFError, OSError):
        shutil.rmtree(batch_dir, ignore_errors=True)
        return False
    return True


def _load_cifar10(
    data_dir: str,
    train: bool,
    transform: transforms.Compose,
) -> datasets.CIFAR10:
    """
    Prefer offline loading. Only uses download=True if the dataset is missing/corrupt.
    """
    try:
        return datasets.CIFAR10(root=data_dir, train=train, download=False, transform=transform)
    except RuntimeError:
        _try_extract_cifar10_tarball(data_dir)
        try:
            return datasets.CIFAR10(root=data_dir, train=train, download=False, transform=transform)
        except RuntimeError:
            return datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)


def _cifar_train_transform_for_lenet() -> transforms.Compose:
    """Light augmentation; images stay 32×32 for LeNet-5."""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def _cifar_eval_transform_for_lenet() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def _cifar_train_transform_for_nasnet(img_size: int, pretrained: bool) -> transforms.Compose:
    """Resize small CIFAR images so NASNet cells operate at a meaningful resolution."""
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if pretrained else (CIFAR_MEAN, CIFAR_STD)
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),  # room for random crop
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _cifar_eval_transform_for_nasnet(img_size: int, pretrained: bool) -> transforms.Compose:
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if pretrained else (CIFAR_MEAN, CIFAR_STD)
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def get_cifar10_loaders(
    model_name: Literal["lenet", "nasnet"],
    batch_size: int,
    img_size: int,
    pretrained: bool,
    num_workers: int,
    seed: int,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    Train/val split: 45k / 5k from the official 50k training images (fixed RNG seed).
    """
    if model_name == "lenet":
        train_tf = _cifar_train_transform_for_lenet()
        eval_tf = _cifar_eval_transform_for_lenet()
    else:
        train_tf = _cifar_train_transform_for_nasnet(img_size, pretrained)
        eval_tf = _cifar_eval_transform_for_nasnet(img_size, pretrained)

    train_set = _load_cifar10(data_dir, train=True, transform=train_tf)
    # Separate transform for validation subset (no random crop / flip).
    val_base = _load_cifar10(data_dir, train=True, transform=eval_tf)
    test_set = _load_cifar10(data_dir, train=False, transform=eval_tf)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_set), generator=g).tolist()
    train_idx = perm[:45_000]
    val_idx = perm[45_000:50_000]

    train_loader = DataLoader(
        Subset(train_set, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(val_base, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader
