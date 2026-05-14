"""
NASNet (Zoph et al., CVPR 2018) — convolutional architecture whose normal cell
and reduction cell were discovered by neural architecture search (NAS), then
stacked manually to form the final network.

Two canonical variants from the same paper:
  - NASNet-A *Mobile*: ~5M params, designed for mobile/edge (the practical one).
  - NASNet-A *Large* : ~85M params, designed for ImageNet accuracy.

For CIFAR-10 on CPU we **default to NASNet-A Mobile** so a full epoch is feasible.
The implementation comes from the `pretrainedmodels` package (Cadene), which is
the reference port of the original TensorFlow weights and code into PyTorch.

NASNet-A Mobile expects 224x224 inputs by default because its head uses a fixed
``AvgPool2d(7)``. We patch that head to ``AdaptiveAvgPool2d(1)`` so the same
network also works at smaller resolutions (e.g. 96 or 128), which is essential
when training from scratch on CPU.
"""

from __future__ import annotations

from typing import Literal

import torch.nn as nn

NASNetVariant = Literal["mobile", "large"]


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
    """NASNet-A Mobile via pretrainedmodels (Cadene)."""
    from pretrainedmodels.models.nasnet_mobile import NASNetAMobile

    if pretrained:
        # Loads ImageNet weights into a 1000-class head, then we swap the head.
        model = NASNetAMobile(num_classes=1000)
        try:
            from pretrainedmodels.models.nasnet_mobile import pretrained_settings
            import torch.utils.model_zoo as model_zoo

            settings = pretrained_settings["nasnetamobile"]["imagenet"]
            state = model_zoo.load_url(settings["url"])
            model.load_state_dict(state, strict=False)
        except Exception as exc:  # pragma: no cover - network or layout issue
            raise RuntimeError(
                "Could not load NASNet-A Mobile ImageNet weights: " + str(exc)
            ) from exc
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
