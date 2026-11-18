"""
Baseline vs. regularized ResNet50 comparison figure.

Reads:
    outputs/results/resnet50_log.csv       (baseline: Adam, wd=1e-4, no dropout, no label smoothing)
    outputs/results/resnet50_reg_log.csv   (regularized: AdamW, wd=5e-4, dropout=0.5, label_smoothing=0.1)

Writes:
    outputs/figures/resnet50_baseline_vs_regularized.png
"""

from pathlib import Path
import csv
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


def read_log(path: Path) -> dict[str, list[float]]:
    cols = {"epoch": [], "train_acc": [], "val_acc": []}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cols["epoch"].append(int(row["epoch"]))
            cols["train_acc"].append(float(row["train_acc"]))
            cols["val_acc"].append(float(row["val_acc"]))
    return cols


baseline = read_log(RESULTS / "resnet50_log.csv")
regularized = read_log(RESULTS / "resnet50_reg_log.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Left: val accuracy comparison ---
ax = axes[0]
ax.plot(baseline["epoch"], baseline["val_acc"], marker="o", ms=5, lw=2,
        color="#c0392b", label="Baseline (Adam, wd=1e-4)")
ax.plot(regularized["epoch"], regularized["val_acc"], marker="s", ms=5, lw=2,
        color="#2980b9", label="Regularized (AdamW, wd=5e-4, dropout=0.5, LS=0.1)")

# Mark best epochs
for run, color in [(baseline, "#c0392b"), (regularized, "#2980b9")]:
    best_idx = run["val_acc"].index(max(run["val_acc"]))
    best_ep = run["epoch"][best_idx]
    best_va = run["val_acc"][best_idx]
    ax.scatter([best_ep], [best_va], s=180, facecolors="none",
               edgecolors=color, linewidths=2, zorder=5)
    ax.annotate(f"best {best_va:.4f}", xy=(best_ep, best_va),
                xytext=(8, 9), textcoords="offset points", color=color,
                fontsize=11)

# Headroom so the "best" labels sit inside the axes instead of on the border.
peak = max(max(baseline["val_acc"]), max(regularized["val_acc"]))
ax.set_ylim(top=peak + 0.006)

ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy on CRC-VAL-HE-7K")
ax.set_title("ResNet50: evaluation accuracy over epochs")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=10)

# --- Right: train - val gap (overfitting indicator) ---
ax = axes[1]
b_gap = [t - v for t, v in zip(baseline["train_acc"], baseline["val_acc"])]
r_gap = [t - v for t, v in zip(regularized["train_acc"], regularized["val_acc"])]
ax.plot(baseline["epoch"], b_gap, marker="o", ms=5, lw=2,
        color="#c0392b", label="Baseline")
ax.plot(regularized["epoch"], r_gap, marker="s", ms=5, lw=2,
        color="#2980b9", label="Regularized")
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Train accuracy − evaluation accuracy")
ax.set_title("ResNet50: generalization gap (higher = more overfitting)")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=11)

plt.suptitle("ResNet50: Baseline vs. Regularized (10 epochs, same data and LR schedule)",
             fontsize=14, y=1.03)
plt.tight_layout()

out_path = FIGURES / "resnet50_baseline_vs_regularized.png"
plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
print(f"Saved: {out_path}")

# --- Console summary ---
def summarize(name, run):
    best_idx = run["val_acc"].index(max(run["val_acc"]))
    print(f"{name:>12}: best val_acc={run['val_acc'][best_idx]:.4f} at epoch {run['epoch'][best_idx]:2d} | "
          f"final train_acc={run['train_acc'][-1]:.4f}, val_acc={run['val_acc'][-1]:.4f} | "
          f"final gap={run['train_acc'][-1] - run['val_acc'][-1]:+.4f}")

print()
summarize("Baseline", baseline)
summarize("Regularized", regularized)
