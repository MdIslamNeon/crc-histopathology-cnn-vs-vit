"""
plot_comparison.py
------------------
Compare multiple training runs from their CSV logs.
Produces:
  - A 1x2 figure: val_acc curves + train/val gap over epochs
  - A summary CSV with best-epoch metrics per run

Usage:
    python plot_comparison.py \
        --runs resnet50 resnet50_reg \
        --labels "Baseline" "Regularized" \
        --title "ResNet50: baseline vs regularized" \
        --out resnet50_baseline_vs_regularized

Output files (under outputs/figures/ and outputs/results/):
    {out}.png
    {out}_summary.csv
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_run(results_dir: Path, run_name: str) -> pd.DataFrame:
    csv_path = results_dir / f"{run_name}_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Log not found: {csv_path}")
    return pd.read_csv(csv_path)


def summarize(df: pd.DataFrame, label: str) -> dict:
    best_row = df.loc[df["val_acc"].idxmax()]
    final_row = df.iloc[-1]
    return {
        "run": label,
        "best_epoch": int(best_row["epoch"]),
        "best_val_acc": round(float(best_row["val_acc"]), 4),
        "final_train_acc": round(float(final_row["train_acc"]), 4),
        "final_val_acc": round(float(final_row["val_acc"]), 4),
        "final_train_val_gap": round(
            float(final_row["train_acc"]) - float(final_row["val_acc"]), 4
        ),
    }


def plot(runs: dict, title: str, out_path: Path) -> None:
    fig, (ax_val, ax_gap) = plt.subplots(1, 2, figsize=(13, 5))

    for label, df in runs.items():
        ax_val.plot(df["epoch"], df["val_acc"], marker="o", label=label)
        gap = df["train_acc"] - df["val_acc"]
        ax_gap.plot(df["epoch"], gap, marker="o", label=label)

        best_idx = df["val_acc"].idxmax()
        ax_val.scatter(
            df["epoch"].iloc[best_idx],
            df["val_acc"].iloc[best_idx],
            marker="*", s=200, zorder=5,
            edgecolors="black", linewidths=0.8,
        )

    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Validation accuracy")
    ax_val.set_title("Validation accuracy (★ = best)")
    ax_val.grid(True, alpha=0.3)
    ax_val.legend()

    ax_gap.set_xlabel("Epoch")
    ax_gap.set_ylabel("Train acc − Val acc")
    ax_gap.set_title("Overfitting gap")
    ax_gap.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax_gap.grid(True, alpha=0.3)
    ax_gap.legend()

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",   nargs="+", required=True,
                        help="Run names (matching {name}_log.csv in outputs/results/)")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Display label per run (same length as --runs)")
    parser.add_argument("--title",  type=str, default="Run comparison")
    parser.add_argument("--out",    type=str, required=True,
                        help="Output filename stem (no extension)")
    parser.add_argument("--output_dir", type=str, default="../outputs")
    args = parser.parse_args()

    if len(args.runs) != len(args.labels):
        raise ValueError("--runs and --labels must have the same length")

    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs = {label: load_run(results_dir, name) for name, label in zip(args.runs, args.labels)}

    summaries = [summarize(df, label) for label, df in runs.items()]

    fig_path = figures_dir / f"{args.out}.png"
    plot(runs, args.title, fig_path)

    summary_path = results_dir / f"{args.out}_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Saved figure : {fig_path}")
    print(f"Saved summary: {summary_path}")
    print()
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
