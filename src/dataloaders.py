from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CRCDataset, TRAIN_MEAN, TRAIN_STD


# -----------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------

def get_train_transform():
    """
    Augmentation pipeline for training.
    Augmentations are applied randomly — each image looks slightly different
    every epoch, which helps prevent overfitting.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
    ])


def get_val_transform():
    """
    Minimal pipeline for validation — no augmentations.
    We want a consistent, unmodified view of val images to measure
    true model performance.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
    ])


# -----------------------------------------------------------------------
# DataLoader factory
# -----------------------------------------------------------------------

def get_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    fraction: float = 1.0,
) -> tuple[DataLoader, DataLoader]:
    """
    Builds and returns (train_loader, val_loader).

    Args:
        data_dir:    Root data directory containing train/ and val/ subfolders.
        batch_size:  Number of images per batch fed to the model.
        num_workers: Parallel workers for loading images from disk.
        fraction:    Fraction of each class to load (for reduced-data experiments).

    Returns:
        (train_loader, val_loader)
    """
    data_dir = Path(data_dir)

    train_dataset = CRCDataset(
        root=data_dir / "train",
        transform=get_train_transform(),
        fraction=fraction,
    )
    val_dataset = CRCDataset(
        root=data_dir / "val",
        transform=get_val_transform(),
        fraction=1.0,  # always evaluate on the full val set
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # randomise order each epoch
        num_workers=num_workers,
        pin_memory=True,       # faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # no need to shuffle for evaluation
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train samples : {len(train_dataset):,}")
    print(f"Val samples   : {len(val_dataset):,}")
    print(f"Batch size    : {batch_size}")
    print(f"Train batches : {len(train_loader):,}")
    print(f"Val batches   : {len(val_loader):,}")

    return train_loader, val_loader
