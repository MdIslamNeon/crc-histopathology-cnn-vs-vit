"""
Why STR and MUS are hard: raw tissue, correct cases next to errors.

No heatmaps here on purpose. The point of this figure is that the two tissues
look alike to a person too, so the committee can judge the difficulty before
being shown what the models attended to.

Selection is by cross-model consensus over the five common-protocol runs.
Correct rows need all five right; error rows need a majority (3+) making the
same mistake, because unanimous MUS->STR errors are too rare to fill a row
(there is exactly one). Within each group, patches are ranked by how many
models agreed and then by confidence, so the errors shown are confident errors
rather than coin flips.

Reads:
    outputs/results/{run}_val_predictions.csv
    data/val/{STR,MUS}/*.tif

Writes:
    outputs/figures/str_mus_examples.png
    outputs/results/str_mus_consensus_examples.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gradcam_compare import RUNS, VAL

DPI = 300
N_COLS = 5

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_predictions():
    """patch -> {model: (predicted, confidence)}, plus patch -> true class."""
    per_patch = defaultdict(dict)
    truth = {}
    for run, _, display in RUNS:
        with open(RESULTS / f"{run}_val_predictions.csv", newline="") as f:
            for row in csv.DictReader(f):
                per_patch[row["patch"]][display] = (row["predicted"],
                                                    float(row["pred_confidence"]))
                truth[row["patch"]] = row["true_class"]
    return per_patch, truth


def consensus(per_patch, truth, true_cls, verdict, min_agree):
    """Patches of true_cls that >=min_agree models call `verdict`.

    Ranked by agreement first, then by how confident the agreeing models were.
    """
    out = []
    for patch, preds in per_patch.items():
        if truth[patch] != true_cls or len(preds) != len(RUNS):
            continue
        agree = [c for p, c in preds.values() if p == verdict]
        if len(agree) >= min_agree:
            out.append((len(agree), float(np.mean(agree)), patch))
    out.sort(reverse=True)
    return out


per_patch, truth = load_predictions()

N = len(RUNS)
ROWS = [
    ("STR", "STR", f"true STR\nall {N} correct", "#1e8449", N),
    ("STR", "MUS", "true STR\nread as MUS", "#c0392b", 3),
    ("MUS", "MUS", f"true MUS\nall {N} correct", "#1e8449", N),
    ("MUS", "STR", "true MUS\nread as STR", "#c0392b", 3),
]

picked = {}
for true_cls, verdict, _, _, min_agree in ROWS:
    picked[(true_cls, verdict)] = consensus(per_patch, truth, true_cls, verdict,
                                            min_agree)
    print(f"{true_cls} called {verdict} by >={min_agree} of {N} models: "
          f"{len(picked[(true_cls, verdict)])} patches")

fig, axes = plt.subplots(len(ROWS), N_COLS, figsize=(3.0 * N_COLS, 3.4 * len(ROWS)))

for i, (true_cls, verdict, label, color, _) in enumerate(ROWS):
    chosen = picked[(true_cls, verdict)][:N_COLS]
    for j in range(N_COLS):
        ax = axes[i, j]
        ax.set_xticks([])
        ax.set_yticks([])
        if j >= len(chosen):
            ax.axis("off")
            continue
        n_agree, conf, patch = chosen[j]
        ax.imshow(Image.open(VAL / true_cls / patch).convert("RGB"))
        ax.set_title(f"{n_agree}/{N} models  p={conf:.2f}",
                     fontsize=14, color=color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
    axes[i, 0].set_ylabel(label, fontsize=17, fontweight="bold", color=color)

fig.suptitle("Stroma and muscle are hard because they look alike\n"
             "red rows are the mistakes — both tissues are pink, fibrous and "
             "aligned at this magnification", fontsize=20, y=0.998)
plt.tight_layout(rect=[0, 0, 1, 0.98])
out = FIGURES / "str_mus_examples.png"
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")

csv_path = RESULTS / "str_mus_consensus_examples.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["true_class", "prediction", "patch", "n_models_agreeing",
                "mean_confidence"])
    for true_cls, verdict, _, _, _ in ROWS:
        for n_agree, conf, patch in picked[(true_cls, verdict)]:
            w.writerow([true_cls, verdict, patch, n_agree, f"{conf:.4f}"])
print(f"Saved: {csv_path}")
