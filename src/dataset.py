from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Tissue classes in consistent order (alphabetical)
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

# Per-channel mean/std computed from the training set (see eda.py)
# These are loaded at import time if available, otherwise fall back to ImageNet values.
_stats_path = Path(__file__).parent.parent / "outputs" / "channel_mean.npy"
if _stats_path.exists():
    TRAIN_MEAN = np.load(_stats_path).tolist()
    TRAIN_STD = np.load(_stats_path.parent / "channel_std.npy").tolist()
else:
    # ImageNet defaults as fallback
    TRAIN_MEAN = [0.485, 0.456, 0.406]
    TRAIN_STD = [0.229, 0.224, 0.225]


class CRCDataset(Dataset):
    """
    PyTorch Dataset for the NCT-CRC-HE-100K histopathology dataset.

    Expects the directory structure:
        root/
            ADI/  *.tif
            BACK/ *.tif
            ...

    Args:
        root:      Path to the split directory (e.g. data/train or data/val).
        transform: torchvision transform applied to each PIL image.
                   If None, only ToTensor + Normalize is applied.
        fraction:  Float in (0, 1]. Load only this fraction of each class
                   (useful for reduced-data experiments in Week 5).
    """

    def __init__(
        self,
        root: str | Path,
        transform=None,
        fraction: float = 1.0,
    ):
        self.root = Path(root)
        self.transform = transform or self._default_transform()
        self.fraction = fraction

        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_samples(self):
        for cls in CLASSES:
            class_dir = self.root / cls
            if not class_dir.is_dir():
                continue
            files = sorted(class_dir.glob("*.tif"))
            if self.fraction < 1.0:
                n = max(1, int(len(files) * self.fraction))
                files = files[:n]
            label = CLASS_TO_IDX[cls]
            self.samples.extend((f, label) for f in files)

    @staticmethod
    def _default_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
        ])

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {cls: 0 for cls in CLASSES}
        for _, label in self.samples:
            counts[CLASSES[label]] += 1
        return counts
