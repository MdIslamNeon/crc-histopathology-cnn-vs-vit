"""
model_factory.py
----------------
Single entry point for loading any architecture used in the thesis benchmark.

Usage:
    from models.model_factory import get_model
    model = get_model("resnet50")
    model = get_model("vit_b16")

Supported names:
    CNN family:
        resnet18, resnet50, resnet101
        densenet121
        efficientnet_b0, efficientnet_b3
        mobilenet_v3

    Transformer family:
        vit_b16            -- Vision Transformer (ViT-B/16)

"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # add src/ to path

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    MobileNet_V3_Large_Weights,
    ViT_B_16_Weights,
)

from dataset import CLASSES

NUM_CLASSES = len(CLASSES)  # 9


def _head(in_features: int, dropout: float) -> nn.Module:
    if dropout > 0:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, NUM_CLASSES))
    return nn.Linear(in_features, NUM_CLASSES)


# -----------------------------------------------------------------------
# Internal builders
# -----------------------------------------------------------------------

def _resnet18(pretrained, dropout):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = _head(m.fc.in_features, dropout)
    return m


def _resnet50(pretrained, dropout):
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    m = models.resnet50(weights=weights)
    m.fc = _head(m.fc.in_features, dropout)
    return m


def _resnet101(pretrained, dropout):
    weights = ResNet101_Weights.DEFAULT if pretrained else None
    m = models.resnet101(weights=weights)
    m.fc = _head(m.fc.in_features, dropout)
    return m


def _densenet121(pretrained, dropout):
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    m = models.densenet121(weights=weights)
    m.classifier = _head(m.classifier.in_features, dropout)
    return m


def _efficientnet_b0(pretrained, dropout):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    m.classifier[1] = _head(m.classifier[1].in_features, dropout)
    return m


def _efficientnet_b3(pretrained, dropout):
    weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_b3(weights=weights)
    m.classifier[1] = _head(m.classifier[1].in_features, dropout)
    return m


def _mobilenet_v3(pretrained, dropout):
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    m = models.mobilenet_v3_large(weights=weights)
    m.classifier[3] = _head(m.classifier[3].in_features, dropout)
    return m


def _vit_b16(pretrained, dropout):
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    m = models.vit_b_16(weights=weights)
    m.heads.head = _head(m.heads.head.in_features, dropout)
    return m


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------

_REGISTRY = {
    "resnet18":        _resnet18,
    "resnet50":        _resnet50,
    "resnet101":       _resnet101,
    "densenet121":     _densenet121,
    "efficientnet_b0": _efficientnet_b0,
    "efficientnet_b3": _efficientnet_b3,
    "mobilenet_v3":    _mobilenet_v3,
    "vit_b16":         _vit_b16,
}


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def get_model(name: str, pretrained: bool = True, dropout: float = 0.0) -> nn.Module:
    """
    Load an architecture by name with the classifier head replaced for
    NUM_CLASSES (9) outputs.

    Args:
        name:       Architecture name (see module docstring for options).
        pretrained: If True, load ImageNet weights. Default True.
        dropout:    If > 0, inserts nn.Dropout(p=dropout) before the final
                    linear layer. Default 0 (no dropout).

    Returns:
        nn.Module ready for training or inference.

    Raises:
        ValueError: If name is not in the registry.
    """
    name = name.lower().strip()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _REGISTRY[name](pretrained, dropout)


def list_models() -> list[str]:
    """Return all registered model names."""
    return sorted(_REGISTRY.keys())


# -----------------------------------------------------------------------
# Quick test
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    dummy = torch.randn(2, 3, 224, 224)

    for name in list_models():
        try:
            model = get_model(name, pretrained=False, dropout=0.5)
            out = model(dummy)
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  {name:<22} output: {tuple(out.shape)}   params: {params:.1f}M")
        except Exception as e:
            print(f"  {name:<22} ERROR: {e}")
