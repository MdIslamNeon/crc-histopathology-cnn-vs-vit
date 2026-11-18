"""
Quantitative Grad-CAM analysis over every STR and MUS validation patch.

Three families of metric, none of which need pixel-level ground truth
(NCT-CRC-HE-100K is labelled per patch only):

  1. Faithfulness  - deletion / insertion curves (Petsiuk et al., RISE).
                     Pixels are replaced with a blurred version of themselves
                     rather than a constant, since flat grey is badly out of
                     distribution for H&E tissue.

                     Curves are reported raw AND normalised by each patch's
                     unperturbed true-class probability. Raw AUC is confounded
                     by calibration: a more confident model starts its deletion
                     curve higher and has more headroom on insertion, which
                     moves both AUCs without its explanation being any better.
                     Normalising measures curve *shape* instead of level.

                     Normalising the start is not enough, because the curves
                     also end at a model-specific floor: the probability the
                     model still assigns to a *fully blurred* patch. That floor
                     ranges from 0.20 (ResNet18) to 0.45 (EfficientNet-B0). A
                     high floor mechanically inflates deletion AUC and inflates
                     insertion AUC too, which is why the two rankings came out
                     near-perfectly reversed - they were measuring blur
                     robustness, not explanation quality.

                     So every curve is paired with a random-ordering control on
                     the same patch, same model, same blur. The reported gap
                     (random - CAM for deletion, CAM - random for insertion,
                     both "higher is better") is what actually isolates whether
                     the heatmap picked informative pixels. Each model is its
                     own baseline, so the floor cancels.

                     The subset is restricted to patches every architecture
                     classifies correctly, so p0 is well away from zero for all
                     models and the comparison uses identical images.

  2. CAM shape     - concentration, entropy and roughness. Quantifies the
                     "ViT speckle vs CNN blob" difference.

  3. Agreement     - pairwise correlation between architectures' heatmaps, at
                     full 224x224 resolution and pooled to ViT's native 14x14
                     grid. The pooled version is the fair one: a high-frequency
                     map correlates poorly with a smooth map almost by
                     construction, so full-resolution agreement understates how
                     much two models actually overlap.

Writes:
    outputs/results/cam_properties.csv
    outputs/results/cam_properties_summary.csv
    outputs/results/cam_faithfulness.csv
    outputs/results/cam_faithfulness_curves.csv
    outputs/results/cam_agreement.csv
    outputs/results/cam_agreement_pooled14.csv
    outputs/figures/cam_faithfulness_curves.png
    outputs/figures/cam_agreement_matrix.png

The CAM-shape figure is drawn separately by plot_cam_properties.py, which reads
cam_properties.csv, so it can be restyled without re-running the GPU pass.
"""

import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import CLASSES, TRAIN_MEAN, TRAIN_STD
from gradcam_compare import RUNS, VAL, FIGURES, RESULTS, cam_config, load_run

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOCUS = ["STR", "MUS"]
CAM_BATCH = 32
DEL_INS_N = 200
STEPS = 50
FWD_BATCH = 26          # STEPS+1 images at once OOMs ViT-B/16 on 8 GB

MEAN_T = torch.tensor(TRAIN_MEAN, device=DEVICE).view(1, 3, 1, 1)
STD_T = torch.tensor(TRAIN_STD, device=DEVICE).view(1, 3, 1, 1)


# ---------------------------------------------------------------- data

def load_focus_patches():
    paths, labels = [], []
    for cls in FOCUS:
        for p in sorted((VAL / cls).glob("*.tif")):
            paths.append(p)
            labels.append(CLASSES.index(cls))
    imgs = np.zeros((len(paths), 224, 224, 3), dtype=np.uint8)
    for i, p in enumerate(tqdm(paths, desc="Loading STR/MUS patches", unit="img")):
        imgs[i] = np.asarray(Image.open(p).convert("RGB"))
    return paths, np.array(labels), imgs


def to_tensor(chunk):
    x = torch.from_numpy(chunk).to(DEVICE).permute(0, 3, 1, 2).float().div_(255.0)
    return (x - MEAN_T) / STD_T


def common_correct(paths):
    """Indices of patches every architecture classifies correctly."""
    sets = []
    for run, _, _ in RUNS:
        s = set()
        with open(RESULTS / f"{run}_val_predictions.csv", newline="") as f:
            for row in csv.DictReader(f):
                if row["correct"] == "1":
                    s.add(row["patch"])
        sets.append(s)
    shared = set.intersection(*sets)
    return np.array([i for i, p in enumerate(paths) if p.name in shared])


# ---------------------------------------------------------------- CAM shape

def cam_shape_metrics(cam):
    flat = cam.ravel().astype(np.float64)
    total = flat.sum() + 1e-12

    k = int(0.1 * flat.size)
    top10 = np.sort(flat)[::-1][:k].sum() / total

    p = flat / total
    entropy = -(p * np.log(p + 1e-12)).sum() / np.log(flat.size)

    dx = np.abs(np.diff(cam, axis=1)).mean()
    dy = np.abs(np.diff(cam, axis=0)).mean()
    return top10, entropy, 0.5 * (dx + dy)


# ---------------------------------------------------------------- faithfulness

def blur_batch(x, k=11):
    pad = k // 2
    return F.avg_pool2d(F.pad(x, (pad,) * 4, mode="reflect"), k, stride=1)


def perturbation_curves(model, img_t, order, cls):
    """Deletion/insertion for one pixel ordering (most important first).

    Vectorised: all STEPS+1 perturbations are built at once instead of looping.
    """
    blurred = blur_batch(img_t.unsqueeze(0))[0]

    rank = np.empty(order.size, dtype=np.int32)
    rank[order] = np.arange(order.size)
    rank_t = torch.from_numpy(rank.reshape(224, 224)).to(DEVICE)

    per = order.size // STEPS
    thresh = (torch.arange(STEPS + 1, device=DEVICE) * per).view(-1, 1, 1, 1)
    mask = rank_t.unsqueeze(0).unsqueeze(0) < thresh          # (STEPS+1,1,224,224)

    img_b = img_t.unsqueeze(0)
    blur_b = blurred.unsqueeze(0)
    del_stack = torch.where(mask, blur_b, img_b)
    ins_stack = torch.where(mask, img_b, blur_b)

    def probs(stack):
        out = []
        with torch.no_grad():
            for s in range(0, len(stack), FWD_BATCH):
                logits = model(stack[s:s + FWD_BATCH])
                out.append(torch.softmax(logits, dim=1)[:, cls].cpu().numpy())
        return np.concatenate(out)

    return probs(del_stack), probs(ins_stack)


# ---------------------------------------------------------------- agreement

def pool14(cams):
    """(n,224,224) -> (n,196), average-pooled onto ViT's native patch grid."""
    x = cams.astype(np.float32).reshape(len(cams), 14, 16, 14, 16)
    return x.mean(axis=(2, 4)).reshape(len(cams), -1)


def correlation_matrix(flat_by_model, names):
    m = np.zeros((len(names), len(names)))
    for a in range(len(names)):
        for b in range(a, len(names)):
            x, y = flat_by_model[names[a]], flat_by_model[names[b]]
            xc = x - x.mean(axis=1, keepdims=True)
            yc = y - y.mean(axis=1, keepdims=True)
            num = (xc * yc).sum(axis=1)
            den = np.sqrt((xc ** 2).sum(axis=1) * (yc ** 2).sum(axis=1)) + 1e-12
            m[a, b] = m[b, a] = (num / den).mean()
    return m


def write_matrix(matrix, names, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + names)
        for a, nm in enumerate(names):
            w.writerow([nm] + [f"{v:.4f}" for v in matrix[a]])


# ---------------------------------------------------------------- main

def main():
    paths, labels, imgs = load_focus_patches()
    n = len(paths)

    pool = common_correct(paths)
    rng = np.random.default_rng(0)
    subset = np.sort(rng.choice(pool, size=min(DEL_INS_N, len(pool)), replace=False))
    print(f"{n} STR/MUS patches; {len(pool)} classified correctly by all "
          f"{len(RUNS)} models; faithfulness uses {len(subset)} of them\n")

    cam_store, rows, curves = {}, [], {}

    for run, arch, display in RUNS:
        print(display)
        model = load_run(run, arch)
        layers, reshape = cam_config(arch, model)

        cams = np.zeros((n, 224, 224), dtype=np.float16)
        preds = np.zeros(n, dtype=np.int64)

        with GradCAM(model=model, target_layers=layers,
                     reshape_transform=reshape) as engine:
            for i in tqdm(range(0, n, CAM_BATCH), desc="  CAMs", leave=False):
                chunk = to_tensor(imgs[i:i + CAM_BATCH])
                lab = labels[i:i + CAM_BATCH]
                with torch.no_grad():
                    preds[i:i + CAM_BATCH] = model(chunk).argmax(1).cpu().numpy()
                targets = [ClassifierOutputTarget(int(c)) for c in lab]
                cams[i:i + CAM_BATCH] = engine(input_tensor=chunk,
                                               targets=targets).astype(np.float16)
        cam_store[display] = cams

        for i in range(n):
            top10, ent, rough = cam_shape_metrics(cams[i].astype(np.float32))
            rows.append([display, paths[i].name, CLASSES[labels[i]],
                         CLASSES[preds[i]], int(preds[i] == labels[i]),
                         f"{top10:.4f}", f"{ent:.4f}", f"{rough:.5f}"])

        acc = {k: [] for k in ("raw_d", "raw_i", "nrm_d", "nrm_i",
                               "rnd_d", "rnd_i")}
        for i in tqdm(subset, desc="  deletion/insertion", leave=False):
            img_t = to_tensor(imgs[i:i + 1])[0]
            cls = int(labels[i])
            cam = cams[i].astype(np.float32)

            cam_order = np.argsort(cam.ravel())[::-1]
            # Seeded per patch, so every architecture is scored against the
            # identical random ordering on the identical image.
            rnd_order = np.random.default_rng(int(i)).permutation(cam.size)

            d, ins = perturbation_curves(model, img_t, cam_order, cls)
            rd, ri = perturbation_curves(model, img_t, rnd_order, cls)

            p0 = max(float(d[0]), 1e-6)
            acc["raw_d"].append(d)
            acc["raw_i"].append(ins)
            acc["nrm_d"].append(d / p0)
            acc["nrm_i"].append(ins / p0)
            acc["rnd_d"].append(rd / p0)
            acc["rnd_i"].append(ri / p0)

        curves[display] = {k: np.mean(v, axis=0) for k, v in acc.items()}

        del model
        torch.cuda.empty_cache()

        dx = 1 / STEPS
        auc = {k: np.trapz(v, dx=dx) for k, v in curves[display].items()}
        print(f"  raw        deletion {auc['raw_d']:.4f}  insertion {auc['raw_i']:.4f}")
        print(f"  normalised deletion {auc['nrm_d']:.4f}  insertion {auc['nrm_i']:.4f}")
        print(f"  random     deletion {auc['rnd_d']:.4f}  insertion {auc['rnd_i']:.4f}")
        print(f"  GAP (higher=better) deletion {auc['rnd_d'] - auc['nrm_d']:+.4f}  "
              f"insertion {auc['nrm_i'] - auc['rnd_i']:+.4f}\n")

    names = [d for _, _, d in RUNS]
    dx = 1 / STEPS

    with open(RESULTS / "cam_properties.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "patch", "true_class", "predicted", "correct",
                    "top10_mass", "entropy", "roughness"])
        w.writerows(rows)

    with open(RESULTS / "cam_properties_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "subset", "n", "top10_mass", "entropy", "roughness"])
        for display in names:
            sub = [r for r in rows if r[0] == display]
            for tag, sel in [("all", sub),
                             ("correct", [r for r in sub if r[4] == 1]),
                             ("incorrect", [r for r in sub if r[4] == 0])]:
                if not sel:
                    continue
                arr = np.array([[float(r[5]), float(r[6]), float(r[7])] for r in sel])
                m = arr.mean(axis=0)
                w.writerow([display, tag, len(sel),
                            f"{m[0]:.4f}", f"{m[1]:.4f}", f"{m[2]:.5f}"])

    aucs = {d: {k: np.trapz(v, dx=dx) for k, v in curves[d].items()} for d in names}

    with open(RESULTS / "cam_faithfulness.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "deletion_auc", "insertion_auc",
                    "deletion_auc_norm", "insertion_auc_norm",
                    "random_deletion_auc_norm", "random_insertion_auc_norm",
                    "deletion_gap", "insertion_gap", "blur_floor", "n_patches"])
        for display in names:
            a = aucs[display]
            w.writerow([display,
                        f"{a['raw_d']:.4f}", f"{a['raw_i']:.4f}",
                        f"{a['nrm_d']:.4f}", f"{a['nrm_i']:.4f}",
                        f"{a['rnd_d']:.4f}", f"{a['rnd_i']:.4f}",
                        f"{a['rnd_d'] - a['nrm_d']:.4f}",
                        f"{a['nrm_i'] - a['rnd_i']:.4f}",
                        f"{curves[display]['nrm_d'][-1]:.4f}", len(subset)])

    with open(RESULTS / "cam_faithfulness_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "step", "fraction", "deletion", "insertion",
                    "deletion_norm", "insertion_norm",
                    "random_deletion_norm", "random_insertion_norm"])
        for display in names:
            c = curves[display]
            for s in range(STEPS + 1):
                w.writerow([display, s, f"{s / STEPS:.3f}",
                            f"{c['raw_d'][s]:.5f}", f"{c['raw_i'][s]:.5f}",
                            f"{c['nrm_d'][s]:.5f}", f"{c['nrm_i'][s]:.5f}",
                            f"{c['rnd_d'][s]:.5f}", f"{c['rnd_i'][s]:.5f}"])

    full = correlation_matrix(
        {d: cam_store[d].reshape(n, -1).astype(np.float32) for d in names}, names)
    pooled = correlation_matrix({d: pool14(cam_store[d]) for d in names}, names)
    write_matrix(full, names, RESULTS / "cam_agreement.csv")
    write_matrix(pooled, names, RESULTS / "cam_agreement_pooled14.csv")

    # ---- figures
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    xs = np.linspace(0, 1, STEPS + 1)
    colors = {d: c for d, c in zip(names, plt.cm.tab10.colors)}
    for display in names:
        c = curves[display]
        axes[0].plot(xs, c["nrm_d"], color=colors[display], label=display)
        axes[0].plot(xs, c["rnd_d"], color=colors[display], ls="--", alpha=0.5)
        axes[1].plot(xs, c["nrm_i"], color=colors[display], label=display)
        axes[1].plot(xs, c["rnd_i"], color=colors[display], ls="--", alpha=0.5)
    axes[0].set_title("Deletion — blur highest-CAM pixels first\n"
                      "(solid below dashed = CAM found the informative pixels)")
    axes[0].set_xlabel("Fraction of pixels removed")
    axes[1].set_title("Insertion — reveal highest-CAM pixels first\n"
                      "(solid above dashed = CAM found the informative pixels)")
    axes[1].set_xlabel("Fraction of pixels revealed")
    for ax in axes[:2]:
        ax.set_ylabel("True-class probability (normalised to unperturbed)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="solid = Grad-CAM order\ndashed = random order",
                  title_fontsize=8)

    y = np.arange(len(names))
    dg = [aucs[d]["rnd_d"] - aucs[d]["nrm_d"] for d in names]
    ig = [aucs[d]["nrm_i"] - aucs[d]["rnd_i"] for d in names]
    axes[2].barh(y - 0.2, dg, 0.4, label="deletion gap", color="#c0392b")
    axes[2].barh(y + 0.2, ig, 0.4, label="insertion gap", color="#1e8449")
    axes[2].set_yticks(y, names)
    axes[2].invert_yaxis()
    axes[2].axvline(0, color="0.3", lw=1)
    axes[2].set_xlabel("AUC gap vs random ordering (higher = more faithful)")
    axes[2].set_title("Faithfulness above chance\n(each model is its own baseline)")
    axes[2].grid(alpha=0.3, axis="x")
    axes[2].legend(fontsize=9)

    plt.suptitle("Grad-CAM faithfulness — patches all architectures classify correctly",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES / "cam_faithfulness_curves.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (m, title) in zip(axes, [(full, "Full resolution (224x224)"),
                                     (pooled, "Pooled to ViT's 14x14 grid")]):
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        for a in range(len(names)):
            for b in range(len(names)):
                ax.text(b, a, f"{m[a, b]:.2f}", ha="center", va="center",
                        color="w" if m[a, b] < 0.6 else "k", fontsize=10)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, label="mean per-patch CAM correlation")
    plt.suptitle("Cross-architecture Grad-CAM agreement", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES / "cam_agreement_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved metrics and figures.")


if __name__ == "__main__":
    main()
