"""
NASNet (Zoph et al., CVPR 2018) — convolutional architecture whose normal cell
and reduction cell were discovered by neural architecture search (NAS), then
stacked manually to form the final network.

Two canonical variants from the same paper:
  - NASNet-A *Mobile*: ~5M params, designed for mobile/edge (the practical one).
  - NASNet-A *Large* : ~85M params, designed for ImageNet accuracy.

For CIFAR-10 on CPU we **default to NASNet-A Mobile** so a full epoch is feasible.
The class implementation comes from ``pretrainedmodels`` (Cadene). ImageNet weights
were originally hosted on ``data.lip6.fr``; that host now serves an **expired TLS
certificate**, so we try that URL first, then fall back to a **GitHub release mirror**
(same ``nasnetamobile-7e03cead`` checkpoint; the ``.pth.tar`` asset is a plain PyTorch
state dict). Override with env ``NASNET_MOBILE_WEIGHT_PATH`` pointing at a local
``.pth`` / checkpoint file to skip all downloads.

NASNet-A Mobile expects 224x224 inputs by default because its head uses a fixed
``AvgPool2d(7)``. We patch that head to ``AdaptiveAvgPool2d(1)`` so the same
network also works at smaller resolutions (e.g. 96 or 128), which is essential
when training from scratch on CPU.
"""

from __future__ import annotations

import os
import warnings
from typing import Literal

import torch
import torch.nn as nn

NASNetVariant = Literal["mobile", "large"]

# Same Cadene NASNet-A Mobile ImageNet checkpoint as ``pretrained_settings``; GitHub TLS is valid.
_GITHUB_MOBILE_STATE_MIRROR = (
    "https://github.com/veronikayurchuk/pretrained-models.pytorch/releases/download/v1.0/"
    "nasnetmobile-7e03cead.pth.tar"
)


def _load_nasnetamobile_imagenet_state_dict(primary_url: str) -> dict:
    """Load the public ImageNet state dict: try primary URL, then TLS-safe GitHub mirror."""
    local = os.environ.get("NASNET_MOBILE_WEIGHT_PATH", "").strip()
    if local:
        path = os.path.expanduser(local)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"NASNET_MOBILE_WEIGHT_PATH is not a file: {path}")
        return torch.load(path, map_location="cpu", weights_only=True)

    try:
        return torch.hub.load_state_dict_from_url(
            primary_url,
            progress=True,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as primary_exc:  # SSL, 404, offline, etc.
        warnings.warn(
            f"NASNet-A Mobile weight download from primary URL failed ({primary_exc!r}); "
            "trying GitHub mirror (same checkpoint, valid certificate).",
            UserWarning,
            stacklevel=2,
        )
    try:
        return torch.hub.load_state_dict_from_url(
            _GITHUB_MOBILE_STATE_MIRROR,
            file_name="nasnetamobile-7e03cead.pth",
            progress=True,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as mirror_exc:
        raise RuntimeError(
            "Could not load NASNet-A Mobile ImageNet weights (primary URL and GitHub mirror). "
            "Download the file manually and set NASNET_MOBILE_WEIGHT_PATH, or fix your network. "
            f"Primary error: {primary_exc!r}; mirror error: {mirror_exc!r}"
        ) from mirror_exc


def _patch_adaptive_pool(model: nn.Module) -> nn.Module:
    """Replace the fixed AvgPool2d head with adaptive pooling so any input size works."""
    if hasattr(model, "avg_pool") and isinstance(model.avg_pool, nn.AvgPool2d):
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
    if hasattr(model, "global_pool"):
        try:
            model.global_pool = nn.AdaptiveAvgPool2d(1)
        except Exception:
            pass
    return model


def _build_nasnet_mobile(num_classes: int, pretrained: bool) -> nn.Module:
    """NASNet-A Mobile via pretrainedmodels (Cadene); pretrained weights via torch.hub URLs."""
    from pretrainedmodels.models.nasnet_mobile import NASNetAMobile, pretrained_settings

    if pretrained:
        model = NASNetAMobile(num_classes=1000)
        settings = pretrained_settings["nasnetamobile"]["imagenet"]
        state = _load_nasnetamobile_imagenet_state_dict(settings["url"])
        model.load_state_dict(state, strict=False)
        in_features = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features, num_classes)
    else:
        model = NASNetAMobile(num_classes=num_classes)

    return _patch_adaptive_pool(model)


def _build_nasnet_large(num_classes: int, pretrained: bool) -> nn.Module:
    """NASNet-A Large via timm (~85M params; mainly for GPU runs)."""
    import timm

    return timm.create_model("nasnetalarge", pretrained=pretrained, num_classes=num_classes)


def build_nasnet(
    num_classes: int = 10,
    pretrained: bool = False,
    variant: NASNetVariant = "mobile",
) -> nn.Module:
    """Build NASNet-A.

    Args:
        num_classes: CIFAR-10 uses 10.
        pretrained: load ImageNet weights (Mobile only currently). Replaces the
            classification head to match ``num_classes``.
        variant: "mobile" (default, ~5M params, CPU-friendly) or
            "large" (~85M params, GPU-recommended).
    """
    if variant == "mobile":
        return _build_nasnet_mobile(num_classes, pretrained)
    if variant == "large":
        return _build_nasnet_large(num_classes, pretrained)
    raise ValueError(f"Unknown NASNet variant: {variant}")


def nasnet_suggested_input_size(pretrained: bool, variant: NASNetVariant = "mobile") -> int:
    """Default spatial size for transforms.

    Mobile from scratch on CPU runs well at 96-128; the head is now adaptive.
    Pretrained / Large evaluations follow the original 224 / 331 recipe.
    """
    if variant == "large":
        return 331 if pretrained else 224
    return 224 if pretrained else 128
