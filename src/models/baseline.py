import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # add src/ to path

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from dataset import CLASSES


def get_baseline_model(num_classes: int = len(CLASSES), pretrained: bool = True) -> nn.Module:
    """
    Returns a ResNet18 model with the final classification layer replaced
    to output num_classes predictions instead of the default 1000 (ImageNet).

    ResNet18 is used as the baseline because:
    - It is small and fast to train, so issues in the pipeline show up quickly
    - ImageNet pretraining gives a strong starting point even for histology images
    - It is well understood, making results easy to interpret

    Args:
        num_classes: Number of output classes (9 for NCT-CRC-HE-100K).
        pretrained:  If True, load ImageNet weights. If False, train from scratch.

    Returns:
        A PyTorch nn.Module ready for training.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace the final fully connected layer
    # Original: Linear(512 -> 1000)  (ImageNet has 1000 classes)
    # Ours:     Linear(512 -> 9)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

if __name__ == "__main__":
    import torch

    model = get_baseline_model()
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)  # 1 image, 3 channels, 224x224
    output = model(dummy)

    print("Output shape:", output.shape)  # Expected: torch.Size([1, 9]) (1 image, 9 scores - one score per class)