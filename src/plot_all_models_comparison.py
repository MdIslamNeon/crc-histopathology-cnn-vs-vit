"""
Cross-architecture comparison across all trained runs.

Writes:
    outputs/figures/all_models_val_accuracy.png
    outputs/figures/all_models_summary.png
    outputs/figures/accuracy_vs_params.png
    outputs/results/all_models_summary.csv
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

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

# (log filename stem, display label, approx param count in millions)
RUNS = [
    ("resnet50",        "ResNet50 (baseline)",    24.0),
    ("resnet50_reg",    "ResNet50 (regularized)", 24.0),
    ("resnet18",        "ResNet18",               11.2),
    ("densenet121",     "DenseNet121",             7.0),
    ("efficientnet_b0", "EfficientNet-B0",         4.0),
    ("vit_b16",         "ViT-B/16",               85.8),
]

COLORS = ["#95a5a6", "#c0392b", "#8e44ad", "#16a085", "#27ae60", "#2980b9"]


def read_log(stem):
    path = RESULTS / f"{stem}_log.csv"
    rows = {"epoch": [], "train_acc": [], "val_acc": []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows["epoch"].append(int(row["epoch"]))
            rows["train_acc"].append(float(row["train_acc"]))
            rows["val_acc"].append(float(row["val_acc"]))
    return rows


data = {stem: read_log(stem) for stem, _, _ in RUNS}

# ---------------- Figure 1: eval accuracy curves ----------------
fig, ax = plt.subplots(figsize=(11, 6.5))
for (stem, label, _), color in zip(RUNS, COLORS):
    r = data[stem]
    style = "--" if stem == "resnet50" else "-"
    ax.plot(r["epoch"], r["val_acc"], marker="o", ms=5, lw=2,
            linestyle=style, color=color, label=label)
    bi = r["val_acc"].index(max(r["val_acc"]))
    ax.scatter([r["epoch"][bi]], [r["val_acc"][bi]], s=170,
               facecolors="none", edgecolors=color, linewidths=2.5, zorder=5)

ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy on CRC-VAL-HE-7K")
ax.set_title("Cross-architecture accuracy over training\n"
             "common training protocol, except ResNet50 (baseline)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
# The peak (circled) is also the epoch that was checkpointed, so it is an
# optimistic estimate - see the caption note on the summary figure.
ax.text(0.01, 0.01, "circles mark each run's peak epoch",
        transform=ax.transAxes, fontsize=10, color="0.35")
plt.tight_layout()
p1 = FIGURES / "all_models_val_accuracy.png"
plt.savefig(p1, dpi=DPI, bbox_inches="tight")
plt.close()

# ---------------- Figure 2: best acc + generalization gap ----------------
color_of = {label: c for (_, label, _), c in zip(RUNS, COLORS)}
best_acc, final_acc, gaps = {}, {}, {}
for stem, label, _ in RUNS:
    r = data[stem]
    best_acc[label] = max(r["val_acc"])
    final_acc[label] = r["val_acc"][-1]
    gaps[label] = r["train_acc"][-1] - r["val_acc"][-1]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# Left: peak vs final epoch. The peak epoch was chosen on this same set, so it
# is an optimistic number; the final epoch is selection-free. Showing both makes
# the size of the selection effect visible instead of hiding it.
names = sorted(best_acc, key=best_acc.get)
y = np.arange(len(names))
ax = axes[0]
ax.barh(y - 0.2, [best_acc[n] for n in names], 0.4,
        color="#2980b9", label="peak epoch")
ax.barh(y + 0.2, [final_acc[n] for n in names], 0.4,
        color="#95a5a6", label="final epoch")
for i, n in enumerate(names):
    ax.text(best_acc[n] + 0.01, i - 0.2, f"{best_acc[n]:.4f}",
            va="center", fontsize=11)
    ax.text(final_acc[n] + 0.01, i + 0.2, f"{final_acc[n]:.4f}",
            va="center", fontsize=11)
# Full 0-1 scale: the point of this panel is that the models are near-identical,
# and a zoomed axis turns a 1.3-point spread into an apparent blowout.
ax.set_xlim(0, 1.15)
ax.set_yticks(y, names)
ax.set_xlabel("Accuracy on CRC-VAL-HE-7K")
ax.set_title("Evaluation accuracy (full scale)")
ax.grid(alpha=0.3, axis="x")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
          fontsize=11, frameon=False)
# The peak epoch was chosen by watching this same set, so it is optimistic; the
# final epoch involves no such choice. The spread between the two is the size of
# the selection effect.
ax.text(0.5, -0.3, "peak epoch was selected on this same set — "
        "final epoch is selection-free",
        transform=ax.transAxes, fontsize=10, color="0.35", ha="center")

# Right: generalization gap, best (smallest) on top.
ax = axes[1]
gap_names = sorted(gaps, key=gaps.get, reverse=True)
bars = ax.barh(gap_names, [gaps[n] for n in gap_names],
               color=[color_of[n] for n in gap_names])
ax.set_xlim(0, 0.072)
for b, n in zip(bars, gap_names):
    ax.text(gaps[n] + 0.0008, b.get_y() + b.get_height() / 2,
            f"{gaps[n]:+.4f}", va="center", fontsize=11)
ax.set_xlabel("Train accuracy − evaluation accuracy (final epoch)")
ax.set_title("Generalization gap (lower = less overfitting)")
ax.grid(alpha=0.3, axis="x")

plt.suptitle("Cross-architecture comparison — NCT-CRC-HE-100K, 10 epochs\n"
             "common training protocol, except ResNet50 (baseline)",
             fontsize=14, y=1.07)
plt.tight_layout()
p2 = FIGURES / "all_models_summary.png"
plt.savefig(p2, dpi=DPI, bbox_inches="tight")
plt.close()

# ---------------- Figure 3: accuracy vs model size ----------------
params_of = {label: p for _, label, p in RUNS}
BASELINE = "ResNet50 (baseline)"

# Nudges in points, chosen so the two ResNet50 runs (same x) do not collide.
LABEL_OFFSET = {
    "EfficientNet-B0":        (0, 11),
    "DenseNet121":            (0, 11),
    "ResNet18":               (0, -18),
    "ResNet50 (regularized)": (11, 6),
    BASELINE:                 (11, -12),
    "ViT-B/16":               (0, 11),
}

fig, ax = plt.subplots(figsize=(9, 5.5))
for label, acc in best_acc.items():
    off = LABEL_OFFSET[label]
    if label == BASELINE:
        ax.scatter(params_of[label], acc, s=130, facecolors="none",
                   edgecolors=color_of[label], linewidths=2, zorder=3)
    else:
        ax.scatter(params_of[label], acc, s=130, color=color_of[label], zorder=3)
    ax.annotate(label, (params_of[label], acc), textcoords="offset points",
                xytext=off, ha="left" if off[0] else "center", fontsize=11)

ax.set_xscale("log")
ax.set_xticks([4, 7, 11.2, 24, 85.8])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlim(3, 130)
ax.set_ylim(0.955, 0.978)
ax.set_xlabel("Parameters (millions, log scale)")
ax.set_ylabel("Peak accuracy on CRC-VAL-HE-7K")
spread = (max(best_acc.values()) - min(best_acc.values())) * 100
ax.set_title("Accuracy does not follow model size\n"
             f"21x more parameters, no gain — full spread is {spread:.1f} "
             "percentage points (hollow marker = old recipe)", fontsize=13)
ax.grid(alpha=0.3)
ax.text(0.01, 0.02, "peak epoch was selected on this same set",
        transform=ax.transAxes, fontsize=10, color="0.35")
plt.tight_layout()
p3 = FIGURES / "accuracy_vs_params.png"
plt.savefig(p3, dpi=DPI, bbox_inches="tight")
plt.close()

# ---------------- Summary CSV ----------------
csv_path = RESULTS / "all_models_summary.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "params_M", "best_val_acc", "best_epoch",
                "final_train_acc", "final_val_acc", "final_gap"])
    for stem, label, params in RUNS:
        r = data[stem]
        bi = r["val_acc"].index(max(r["val_acc"]))
        w.writerow([label, params, f"{r['val_acc'][bi]:.4f}", r["epoch"][bi],
                    f"{r['train_acc'][-1]:.4f}", f"{r['val_acc'][-1]:.4f}",
                    f"{r['train_acc'][-1] - r['val_acc'][-1]:+.4f}"])

print(f"Saved: {p1}")
print(f"Saved: {p2}")
print(f"Saved: {p3}")
print(f"Saved: {csv_path}\n")

print(f"{'Model':<24}{'Params':>8}{'Best val':>10}{'@ep':>5}{'Final gap':>11}")
print("-" * 58)
for stem, label, params in RUNS:
    r = data[stem]
    bi = r["val_acc"].index(max(r["val_acc"]))
    print(f"{label:<24}{params:>7.1f}M{max(r['val_acc']):>10.4f}{r['epoch'][bi]:>5}"
          f"{r['train_acc'][-1] - r['val_acc'][-1]:>+11.4f}")
