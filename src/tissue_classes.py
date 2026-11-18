import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "data" / "train"
FIGURES = PROJECT_ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

CLASSES = ["ADI","BACK","DEB","LYM","MUC","MUS","NORM","STR","TUM"]
NAMES = {
    "ADI":"Adipose", "BACK":"Background", "DEB":"Debris",
    "LYM":"Lymphocytes", "MUC":"Mucus", "MUS":"Smooth muscle",
    "NORM":"Normal mucosa", "STR":"Stroma", "TUM":"Tumor",
}

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for ax, c in zip(axes.ravel(), CLASSES):
    folder = os.path.join(ROOT, c)
    fname = sorted(os.listdir(folder))[0]        # first patch in the class
    ax.imshow(Image.open(os.path.join(folder, fname)))
    ax.set_title(f"{c} — {NAMES[c]}", fontsize=11)
    ax.axis("off")

plt.tight_layout()
plt.savefig(FIGURES / "tissue_classes.png", dpi=200, bbox_inches="tight")