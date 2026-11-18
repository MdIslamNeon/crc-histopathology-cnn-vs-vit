from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Train dir exists:", TRAIN_DIR.exists(), "->", TRAIN_DIR.resolve())
print("Val dir exists:", VAL_DIR.exists(), "->", VAL_DIR.resolve())


# -------- CLASS LISTING --------

def list_classes(base_dir: Path) -> list[str]:
    return sorted([p.name for p in base_dir.iterdir() if p.is_dir()])

train_classes = list_classes(TRAIN_DIR)
val_classes = list_classes(VAL_DIR)

print("\nTrain classes:", train_classes)
print("Val classes:", val_classes)
print("\nSame classes?", train_classes == val_classes)


# -------- COUNT IMAGES --------

def count_images(base_dir: Path) -> dict[str, int]:
    counts = {}
    for class_dir in base_dir.iterdir():
        if class_dir.is_dir():
            counts[class_dir.name] = sum(1 for f in class_dir.iterdir() if f.is_file())
    return counts

train_counts = count_images(TRAIN_DIR)
val_counts = count_images(VAL_DIR)

print("\nTrain counts:")
for cls in sorted(train_counts):
    print(f"  {cls}: {train_counts[cls]}")

print("\nVal counts:")
for cls in sorted(val_counts):
    print(f"  {cls}: {val_counts[cls]}")

print(f"\nTotal train images: {sum(train_counts.values())}")
print(f"Total val images:   {sum(val_counts.values())}")


# -------- CLASS DISTRIBUTION BAR CHART --------

def plot_class_distribution(train_counts: dict, val_counts: dict, save_path: Path):
    classes = sorted(train_counts.keys())
    x = np.arange(len(classes))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, [train_counts[c] for c in classes], width, label="Train", color="steelblue")
    ax.bar(x + width / 2, [val_counts[c] for c in classes], width, label="Val", color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution — NCT-CRC-HE-100K")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved class distribution chart -> {save_path}")

plot_class_distribution(train_counts, val_counts, OUTPUT_DIR / "class_distribution.png")


# -------- SAMPLE IMAGES GRID --------

def plot_sample_images(base_dir: Path, classes: list[str], save_path: Path, n_samples: int = 3):
    fig, axes = plt.subplots(len(classes), n_samples, figsize=(n_samples * 2.5, len(classes) * 2.5))
    fig.suptitle(f"Sample Images — {base_dir.name}", fontsize=14)

    for row, cls in enumerate(classes):
        class_dir = base_dir / cls
        images = sorted(class_dir.iterdir())[:n_samples]
        for col, img_path in enumerate(images):
            img = Image.open(img_path).convert("RGB")
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(cls, fontsize=9, rotation=0, labelpad=40, va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved sample images grid -> {save_path}")

plot_sample_images(TRAIN_DIR, train_classes, OUTPUT_DIR / "sample_images.png")


# -------- CHANNEL STATISTICS (mean & std per channel) --------

def compute_channel_stats(base_dir: Path, max_per_class: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimates per-channel mean and std over the dataset.
    Samples up to max_per_class images per class for speed.
    Returns mean and std as arrays of shape (3,) in [0, 1] range.
    """
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for class_dir in sorted(base_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        files = sorted(class_dir.iterdir())[:max_per_class]
        for img_path in files:
            try:
                img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
                pixel_sum += img.reshape(-1, 3).sum(axis=0)
                pixel_sq_sum += (img ** 2).reshape(-1, 3).sum(axis=0)
                pixel_count += img.shape[0] * img.shape[1]
            except Exception:
                continue

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)
    return mean, std

print("\nComputing channel statistics (sampling up to 200 images/class)...")
mean, std = compute_channel_stats(TRAIN_DIR)
print(f"  Mean (R, G, B): {mean.round(4)}")
print(f"  Std  (R, G, B): {std.round(4)}")

np.save(OUTPUT_DIR / "channel_mean.npy", mean)
np.save(OUTPUT_DIR / "channel_std.npy", std)
print(f"Saved channel_mean.npy and channel_std.npy -> {OUTPUT_DIR}/")


# -------- CORRUPT FILE CHECK --------

def find_corrupt_files(base_dir: Path) -> list[Path]:
    corrupt = []
    for img_path in base_dir.rglob("*.tif"):
        try:
            with Image.open(img_path) as img:
                img.verify()
        except (UnidentifiedImageError, Exception):
            corrupt.append(img_path)
    return corrupt

print("\nChecking for corrupt .tif files in train and val...")
corrupt_train = find_corrupt_files(TRAIN_DIR)
corrupt_val = find_corrupt_files(VAL_DIR)

if corrupt_train:
    print(f"  Corrupt train files ({len(corrupt_train)}):")
    for f in corrupt_train:
        print(f"    {f}")
else:
    print("  No corrupt files found in train.")

if corrupt_val:
    print(f"  Corrupt val files ({len(corrupt_val)}):")
    for f in corrupt_val:
        print(f"    {f}")
else:
    print("  No corrupt files found in val.")
