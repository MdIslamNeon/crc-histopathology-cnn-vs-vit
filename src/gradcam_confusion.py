"""
Grad-CAM on the STR<->MUS patches the models actually get wrong.

Selection is driven by cross-model consensus: patches ranked by how many
architectures independently make the same mistake, so the figures show
systematically hard tissue rather than one model's idiosyncratic failure.

Writes:
    outputs/figures/gradcam_confusion_STR_as_MUS.png
    outputs/figures/gradcam_confusion_MUS_as_STR.png
    outputs/figures/gradcam_contrastive_STR_vs_MUS.png
    outputs/results/consensus_hard_cases.csv
"""

import csv
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import CLASSES
from gradcam_compare import (
    RUNS, VAL, FIGURES, RESULTS, DPI, FS_ROW, FS_TILE, FS_CAPTION,
    FS_SUPTITLE, cam_config, load_batch, load_run,
)

warnings.filterwarnings("ignore")

N_SAMPLES = 4


def load_predictions():
    per_patch = defaultdict(dict)
    truth = {}
    for run, _, display in RUNS:
        with open(RESULTS / f"{run}_val_predictions.csv", newline="") as f:
            for row in csv.DictReader(f):
                per_patch[row["patch"]][display] = row["predicted"]
                truth[row["patch"]] = row["true_class"]
    return per_patch, truth


def consensus_cases(per_patch, truth, true_cls, wrong_cls):
    """Patches of true_cls ranked by how many models call them wrong_cls."""
    ranked = []
    for patch, preds in per_patch.items():
        if truth[patch] != true_cls:
            continue
        n = sum(1 for p in preds.values() if p == wrong_cls)
        if n:
            ranked.append((n, patch))
    ranked.sort(key=lambda r: (-r[0], r[1]))
    return ranked


def cam_for(model, arch, batch, class_indices):
    layers, reshape = cam_config(arch, model)
    targets = [ClassifierOutputTarget(int(c)) for c in class_indices]
    with GradCAM(model=model, target_layers=layers, reshape_transform=reshape) as cam:
        return cam(input_tensor=batch, targets=targets)


def confusion_figure(true_cls, wrong_cls, ranked, out_path):
    picks = [p for _, p in ranked[:N_SAMPLES]]
    counts = {p: n for n, p in ranked[:N_SAMPLES]}
    paths = [VAL / true_cls / p for p in picks]
    rgb, batch = load_batch(paths)

    n_rows = len(RUNS) + 1
    fig, axes = plt.subplots(n_rows, len(picks),
                             figsize=(3.1 * len(picks), 3.3 * n_rows))

    for j, p in enumerate(picks):
        ax = axes[0, j]
        ax.imshow(rgb[j])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{counts[p]}/{len(RUNS)} models say {wrong_cls}",
                     fontsize=FS_CAPTION, color="0.25")
    axes[0, 0].set_ylabel("Original", fontsize=FS_ROW, fontweight="bold")

    for i, (run, arch, display) in enumerate(RUNS, start=1):
        model = load_run(run, arch)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        heatmaps = cam_for(model, arch, batch, preds)

        for j in range(len(picks)):
            ax = axes[i, j]
            ax.imshow(show_cam_on_image(rgb[j], heatmaps[j], use_rgb=True))
            ax.set_xticks([])
            ax.set_yticks([])
            correct = CLASSES[preds[j]] == true_cls
            ax.set_title(f"{CLASSES[preds[j]]}  p={probs[j][preds[j]]:.2f}",
                         fontsize=FS_TILE, fontweight="bold",
                         color="#1e8449" if correct else "#c0392b")
        axes[i, 0].set_ylabel(display, fontsize=FS_ROW, fontweight="bold")

        del model
        torch.cuda.empty_cache()

    fig.suptitle(
        f"Grad-CAM on hard {true_cls} patches — heatmap targets each model's OWN prediction\n"
        f"(these are the patches most often misread as {wrong_cls})",
        fontsize=FS_SUPTITLE, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def contrastive_figure(true_cls, wrong_cls, patch, out_path):
    """One patch, both hypotheses: where is the evidence for STR vs for MUS?"""
    paths = [VAL / true_cls / patch]
    rgb, batch = load_batch(paths)
    i_true, i_wrong = CLASSES.index(true_cls), CLASSES.index(wrong_cls)

    fig, axes = plt.subplots(len(RUNS), 3, figsize=(11, 3.6 * len(RUNS)))

    for i, (run, arch, display) in enumerate(RUNS):
        model = load_run(run, arch)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()[0]
        cam_true = cam_for(model, arch, batch, [i_true])[0]
        cam_wrong = cam_for(model, arch, batch, [i_wrong])[0]

        for col, (img, title, color) in enumerate([
            (rgb[0], "Original", "0.25"),
            (show_cam_on_image(rgb[0], cam_true, use_rgb=True),
             f"evidence for {true_cls}  p={probs[i_true]:.2f}", "#1e8449"),
            (show_cam_on_image(rgb[0], cam_wrong, use_rgb=True),
             f"evidence for {wrong_cls}  p={probs[i_wrong]:.2f}", "#c0392b"),
        ]):
            ax = axes[i, col]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=FS_TILE, fontweight="bold", color=color)
        axes[i, 0].set_ylabel(display, fontsize=FS_ROW, fontweight="bold")

        del model
        torch.cuda.empty_cache()

    fig.suptitle(
        f"Competing evidence on a single hard patch (true class {true_cls})\n{patch}",
        fontsize=FS_SUPTITLE, y=0.997)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def dissent_table(per_patch, truth):
    """How often is each model the lone voice of reason on a hard STR/MUS patch?

    A 'unique rescue' is a patch every other architecture misreads as the
    opposing class while this one model gets it right.
    """
    pair = {"STR": "MUS", "MUS": "STR"}
    rescues = {display: 0 for _, _, display in RUNS}
    lone_errors = {display: 0 for _, _, display in RUNS}

    for patch, preds in per_patch.items():
        true_cls = truth[patch]
        if true_cls not in pair:
            continue
        other = pair[true_cls]
        wrong = [m for m, p in preds.items() if p == other]
        right = [m for m, p in preds.items() if p == true_cls]
        if len(wrong) == len(RUNS) - 1 and len(right) == 1:
            rescues[right[0]] += 1
        if len(wrong) == 1 and len(right) == len(RUNS) - 1:
            lone_errors[wrong[0]] += 1

    out = RESULTS / "strmus_dissent.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "unique_rescues", "lone_errors"])
        for _, _, display in RUNS:
            w.writerow([display, rescues[display], lone_errors[display]])
    print(f"Saved: {out}")

    print(f"\n{'Model':<20}{'unique rescues':>16}{'lone errors':>14}")
    for _, _, display in RUNS:
        print(f"{display:<20}{rescues[display]:>16}{lone_errors[display]:>14}")
    print()
    return rescues, lone_errors


def main():
    per_patch, truth = load_predictions()

    str_as_mus = consensus_cases(per_patch, truth, "STR", "MUS")
    mus_as_str = consensus_cases(per_patch, truth, "MUS", "STR")
    dissent_table(per_patch, truth)

    n_models = len(RUNS)
    for name, ranked in [("STR->MUS", str_as_mus), ("MUS->STR", mus_as_str)]:
        unanimous = sum(1 for n, _ in ranked if n == n_models)
        print(f"{name}: {len(ranked)} patches missed by >=1 model, "
              f"{unanimous} missed by all {n_models}")

    with open(RESULTS / "consensus_hard_cases.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_class", "confused_with", "patch", "n_models_wrong",
                    "per_model_predictions"])
        for true_cls, wrong_cls, ranked in [("STR", "MUS", str_as_mus),
                                            ("MUS", "STR", mus_as_str)]:
            for n, patch in ranked:
                preds = ";".join(f"{m}={p}" for m, p in per_patch[patch].items())
                w.writerow([true_cls, wrong_cls, patch, n, preds])
    print(f"Saved: {RESULTS / 'consensus_hard_cases.csv'}")

    confusion_figure("STR", "MUS", str_as_mus,
                     FIGURES / "gradcam_confusion_STR_as_MUS.png")
    confusion_figure("MUS", "STR", mus_as_str,
                     FIGURES / "gradcam_confusion_MUS_as_STR.png")
    contrastive_figure("STR", "MUS", str_as_mus[0][1],
                       FIGURES / "gradcam_contrastive_STR_vs_MUS.png")


if __name__ == "__main__":
    main()
