"""
Cross-architecture Grad-CAM comparison, focused on the STR / MUS confusion pair.

Writes:
    outputs/figures/gradcam_STR.png
    outputs/figures/gradcam_MUS.png
    outputs/results/gradcam_predictions.csv
"""

import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import CLASSES, TRAIN_MEAN, TRAIN_STD
from models.model_factory import get_model

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "data" / "val"
CKPT = ROOT / "outputs" / "checkpoints"
FIGURES = ROOT / "outputs" / "figures"
RESULTS = ROOT / "outputs" / "results"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 4
FOCUS = ["STR", "MUS"]

# These panels are ~12x20 inches, so point sizes that look fine in a notebook
# shrink to nothing once the image is scaled onto a slide. Sized for projection.
DPI = 300
FS_ROW = 20      # architecture name down the left edge
FS_TILE = 17     # per-tile prediction
FS_CAPTION = 13  # patch id above the originals
FS_SUPTITLE = 22

# (checkpoint dir, architecture name for get_model, display label)
RUNS = [
    ("resnet18",        "resnet18",        "ResNet18"),
    ("resnet50_reg",    "resnet50",        "ResNet50 (reg)"),
    ("densenet121",     "densenet121",     "DenseNet121"),
    ("efficientnet_b0", "efficientnet_b0", "EfficientNet-B0"),
    ("vit_b16",         "vit_b16",         "ViT-B/16"),
]

MEAN = np.array(TRAIN_MEAN, dtype=np.float32)
STD = np.array(TRAIN_STD, dtype=np.float32)


def vit_reshape(tensor):
    """torchvision ViT-B/16 token sequence -> CNN-style feature map.

    Index 0 is the class token and has no spatial position, so it is dropped
    before folding the remaining 196 patch tokens back into a 14x14 grid.
    """
    patches = tensor[:, 1:, :]
    b, _, c = patches.shape
    return patches.reshape(b, 14, 14, c).permute(0, 3, 1, 2)


def cam_config(arch, model):
    if arch.startswith("resnet"):
        return [model.layer4[-1]], None
    if arch.startswith("densenet"):
        return [model.features[-1]], None
    if arch.startswith("efficientnet"):
        return [model.features[-1]], None
    if arch == "vit_b16":
        return [model.encoder.layers[-1].ln_1], vit_reshape
    raise ValueError(f"No Grad-CAM target layer defined for '{arch}'")


def load_run(run_name, arch):
    """Load a checkpoint, tolerating the head shape of pre-dropout runs."""
    state = torch.load(CKPT / run_name / "best.pth", map_location="cpu")["model_state_dict"]
    for dropout in (0.5, 0.0):
        model = get_model(arch, pretrained=False, dropout=dropout)
        try:
            model.load_state_dict(state)
        except RuntimeError:
            continue
        return model.to(DEVICE).eval()
    raise RuntimeError(f"Checkpoint for '{run_name}' matches no known head layout")


def pick_samples(cls, n):
    files = sorted((VAL / cls).glob("*.tif"))
    step = max(1, len(files) // n)
    return files[::step][:n]


def load_batch(paths):
    rgb = np.stack([
        np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        for p in paths
    ])
    tensor = torch.from_numpy((rgb - MEAN) / STD).permute(0, 3, 1, 2).contiguous()
    return rgb, tensor.to(DEVICE)


def main():
    pred_rows = []

    for cls in FOCUS:
        paths = pick_samples(cls, N_SAMPLES)
        rgb, batch = load_batch(paths)
        label = CLASSES.index(cls)
        targets = [ClassifierOutputTarget(label)] * len(paths)

        n_rows = len(RUNS) + 1
        fig, axes = plt.subplots(n_rows, N_SAMPLES,
                                 figsize=(3.1 * N_SAMPLES, 3.3 * n_rows))

        for j, p in enumerate(paths):
            ax = axes[0, j]
            ax.imshow(rgb[j])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(p.name.replace(".tif", "")[-14:],
                         fontsize=FS_CAPTION, color="0.35")
        axes[0, 0].set_ylabel("Original", fontsize=FS_ROW, fontweight="bold")

        for i, (run, arch, display) in enumerate(RUNS, start=1):
            model = load_run(run, arch)
            layers, reshape = cam_config(arch, model)

            with torch.no_grad():
                probs = torch.softmax(model(batch), dim=1).cpu().numpy()

            with GradCAM(model=model, target_layers=layers,
                         reshape_transform=reshape) as cam:
                heatmaps = cam(input_tensor=batch, targets=targets)

            for j in range(N_SAMPLES):
                ax = axes[i, j]
                ax.imshow(show_cam_on_image(rgb[j], heatmaps[j], use_rgb=True))
                ax.set_xticks([])
                ax.set_yticks([])
                pred = int(probs[j].argmax())
                correct = pred == label
                ax.set_title(f"{CLASSES[pred]}  p={probs[j][pred]:.2f}",
                             fontsize=FS_TILE, fontweight="bold",
                             color="#1e8449" if correct else "#c0392b")
                pred_rows.append([cls, paths[j].name, display, CLASSES[pred],
                                  f"{probs[j][pred]:.4f}",
                                  f"{probs[j][label]:.4f}", int(correct)])

            axes[i, 0].set_ylabel(display, fontsize=FS_ROW, fontweight="bold")

            del model
            torch.cuda.empty_cache()

        fig.suptitle(
            f"Grad-CAM across architectures — true class: {cls}\n"
            f"(heatmap shows evidence for {cls}; title shows each model's own prediction)",
            fontsize=FS_SUPTITLE, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.985])
        out = FIGURES / f"gradcam_{cls}.png"
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")

    csv_path = RESULTS / "gradcam_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_class", "patch", "model", "predicted",
                    "pred_confidence", "true_class_prob", "correct"])
        w.writerows(pred_rows)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
