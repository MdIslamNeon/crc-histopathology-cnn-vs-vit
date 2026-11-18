"""
Grad-CAM shape on correctly vs incorrectly classified STR/MUS patches.

Reads outputs/results/cam_properties.csv (written by gradcam_metrics.py).

Writes:
    outputs/figures/cam_properties.png
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

# Each panel carries a plain-language gloss: the committee should not have to
# infer what "entropy" means from the axis alone.
PANELS = [
    ("top10_mass", "Concentration",
     "share of heat in the hottest 10% of the map\nhigher = narrower focus",
     "{:.3f}"),
    ("entropy", "Spread",
     "how evenly the heat is spread out\nhigher = the model looks everywhere",
     "{:.3f}"),
    ("roughness", "Speckle",
     "how much the map jumps pixel to pixel\nhigher = patchy, less coherent",
     "{:.4f}"),
]

rows = []
with open(RESULTS / "cam_properties.csv", newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

# dict preserves insertion order, which is the RUNS order gradcam_metrics used
names = list(dict.fromkeys(r["model"] for r in rows))


def mean_of(model, field, correct):
    vals = [float(r[field]) for r in rows
            if r["model"] == model and int(r["correct"]) == correct]
    return np.mean(vals)


fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
y = np.arange(len(names))

for ax, (field, label, gloss, fmt) in zip(axes, PANELS):
    ok = [mean_of(m, field, 1) for m in names]
    bad = [mean_of(m, field, 0) for m in names]
    ax.barh(y - 0.2, ok, 0.4, label="classified correctly", color="#2980b9")
    ax.barh(y + 0.2, bad, 0.4, label="classified incorrectly", color="#c0392b")
    for i, (a, b) in enumerate(zip(ok, bad)):
        ax.text(a, i - 0.2, f" {fmt.format(a)}", va="center", fontsize=11)
        ax.text(b, i + 0.2, f" {fmt.format(b)}", va="center", fontsize=11)
    ax.set_xlim(0, max(max(ok), max(bad)) * 1.3)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_title(f"{label}\n", fontweight="bold")
    ax.set_xlabel(gloss, fontsize=11, color="0.35")
    ax.grid(alpha=0.3, axis="x")

fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.07), frameon=False)
plt.suptitle("When a model is wrong, its Grad-CAM narrows — 5 of 5 architectures\n"
             "STR/MUS patches: errors carry more concentrated, lower-entropy, "
             "smoother heatmaps", fontsize=15)
plt.tight_layout()
out = FIGURES / "cam_properties.png"
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()

print(f"Saved: {out}\n")
print(f"{'Model':<18}{'metric':<12}{'correct':>9}{'incorrect':>11}{'change':>9}")
print("-" * 59)
for m in names:
    for field, _, _, fmt in PANELS:
        a, b = mean_of(m, field, 1), mean_of(m, field, 0)
        print(f"{m:<18}{field:<12}{fmt.format(a):>9}{fmt.format(b):>11}"
              f"{(b - a) / a * 100:>+8.1f}%")
