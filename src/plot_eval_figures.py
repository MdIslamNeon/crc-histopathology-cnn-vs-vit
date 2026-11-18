"""
Confusion matrices and per-class F1, drawn from the CSVs evaluate.py wrote.

Reads:
    outputs/results/{run}_val_predictions.csv
    outputs/results/all_models_per_class_f1.csv

Writes:
    outputs/figures/{run}_confusion_matrix.png
    outputs/figures/all_models_per_class_f1.png

Kept separate from evaluate.py so the figures can be restyled without a GPU
pass over the 7,180-image evaluation set.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dataset import CLASSES
from evaluate import RUNS

# Sized for projection: these figures are read from the back of a room.
DPI = 300
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
})

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

IDX = {c: i for i, c in enumerate(CLASSES)}


def confusion_from_csv(run):
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    with open(RESULTS / f"{run}_val_predictions.csv", newline="") as f:
        for row in csv.DictReader(f):
            cm[IDX[row["true_class"]], IDX[row["predicted"]]] += 1
    return cm


def plot_confusion(cm, display, save_path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                annot_kws={"fontsize": 12},
                cbar_kws={"label": "% of actual class"}, vmin=0, vmax=100)
    # Outline the STR/MUS block — the confusion the thesis is about.
    for i, j in [(IDX["STR"], IDX["MUS"]), (IDX["MUS"], IDX["STR"])]:
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                   edgecolor="#c0392b", lw=3))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{display} — confusion matrix (% of actual class)\n"
                 "outlined cells: the STR/MUS confusion")
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_f1():
    f1_table = {}
    with open(RESULTS / "all_models_per_class_f1.csv", newline="") as f:
        for row in csv.DictReader(f):
            f1_table[row["model"]] = [float(row[c]) for c in CLASSES]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    width = 0.8 / len(f1_table)
    x = np.arange(len(CLASSES))
    for k, (display, f1s) in enumerate(f1_table.items()):
        ax.bar(x + k * width - 0.4 + width / 2, f1s, width, label=display)
    # Shade the two classes that account for most of the remaining errors.
    for c in ("MUS", "STR"):
        ax.axvspan(IDX[c] - 0.5, IDX[c] + 0.5, color="#c0392b", alpha=0.07)
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0.70, 1.005)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 across architectures — CRC-VAL-HE-7K\n"
                 "every architecture bottoms out on the same two classes")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.07),
              fontsize=11, frameon=False)
    plt.tight_layout()
    out = FIGURES / "all_models_per_class_f1.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    return out


for run, _, display in RUNS:
    cm = confusion_from_csv(run)
    out = FIGURES / f"{run}_confusion_matrix.png"
    plot_confusion(cm, display, out)
    strmus = cm[IDX["STR"], IDX["MUS"]] / cm[IDX["STR"]].sum() * 100
    musstr = cm[IDX["MUS"], IDX["STR"]] / cm[IDX["MUS"]].sum() * 100
    errors = cm.sum() - np.trace(cm)
    share = (cm[IDX["STR"], IDX["MUS"]] + cm[IDX["MUS"], IDX["STR"]]) / errors * 100
    print(f"{display:<22} STR->MUS {strmus:5.1f}%  MUS->STR {musstr:5.1f}%  "
          f"= {share:4.1f}% of all errors")
    print(f"  Saved: {out}")

print(f"\nSaved: {plot_f1()}")
