"""
Full-validation-set evaluation for every trained run.

Per run:
    outputs/results/{run}_per_class_metrics.csv
    outputs/results/{run}_val_predictions.csv     <- feeds the Grad-CAM confusion analysis

Across runs:
    outputs/results/all_models_per_class_f1.csv
    outputs/results/confusion_pairs.csv

The confusion-matrix and per-class-F1 figures are drawn by plot_eval_figures.py,
which reads these CSVs, so they can be restyled without re-running the GPU pass.
"""

import csv
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from dataset import CLASSES, TRAIN_MEAN, TRAIN_STD, CRCDataset
from models.model_factory import get_model

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CKPT = ROOT / "outputs" / "checkpoints"
FIGURES = ROOT / "outputs" / "figures"
RESULTS = ROOT / "outputs" / "results"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 64

# (checkpoint dir, architecture for get_model, display label)
RUNS = [
    ("resnet18",        "resnet18",        "ResNet18"),
    ("resnet50",        "resnet50",        "ResNet50 (baseline)"),
    ("resnet50_reg",    "resnet50",        "ResNet50 (reg)"),
    ("densenet121",     "densenet121",     "DenseNet121"),
    ("efficientnet_b0", "efficientnet_b0", "EfficientNet-B0"),
    ("vit_b16",         "vit_b16",         "ViT-B/16"),
]


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


def load_val_cache():
    """Decode the whole val split once into uint8 (~1.1 GB) and reuse across runs."""
    ds = CRCDataset(root=DATA / "val")
    paths = [p for p, _ in ds.samples]
    labels = np.array([l for _, l in ds.samples], dtype=np.int64)
    imgs = np.zeros((len(paths), 224, 224, 3), dtype=np.uint8)
    for i, p in enumerate(tqdm(paths, desc="Caching val images", unit="img")):
        imgs[i] = np.asarray(Image.open(p).convert("RGB"))
    return paths, labels, imgs


MEAN_T = torch.tensor(TRAIN_MEAN, device=DEVICE).view(1, 3, 1, 1)
STD_T = torch.tensor(TRAIN_STD, device=DEVICE).view(1, 3, 1, 1)


def predict(model, imgs, label):
    preds, confs, true_probs = [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(imgs), BATCH), desc="  inference", leave=False):
            chunk = torch.from_numpy(imgs[i:i + BATCH]).to(DEVICE, non_blocking=True)
            x = chunk.permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - MEAN_T) / STD_T
            probs = torch.softmax(model(x), dim=1)
            conf, idx = probs.max(dim=1)
            preds.append(idx.cpu().numpy())
            confs.append(conf.cpu().numpy())
            rows = torch.arange(len(chunk), device=DEVICE)
            true_probs.append(probs[rows, torch.from_numpy(label[i:i + BATCH]).to(DEVICE)].cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs), np.concatenate(true_probs)


def main():
    paths, labels, imgs = load_val_cache()
    print(f"Cached {len(paths):,} val images ({imgs.nbytes / 1e9:.2f} GB)\n")

    f1_table = {}
    pair_rows = []

    for run, arch, display in RUNS:
        print(f"{display}")
        model = load_run(run, arch)
        preds, confs, true_probs = predict(model, imgs, labels)
        del model
        torch.cuda.empty_cache()

        acc = (preds == labels).mean()
        print(f"  overall val accuracy: {acc:.4f}")

        report = classification_report(labels, preds, target_names=CLASSES,
                                       digits=4, output_dict=True, zero_division=0)
        with open(RESULTS / f"{run}_per_class_metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "precision", "recall", "f1", "support"])
            for c in CLASSES:
                r = report[c]
                w.writerow([c, f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                            f"{r['f1-score']:.4f}", int(r["support"])])
            w.writerow(["macro avg", f"{report['macro avg']['precision']:.4f}",
                        f"{report['macro avg']['recall']:.4f}",
                        f"{report['macro avg']['f1-score']:.4f}", len(labels)])
            w.writerow(["accuracy", "", "", f"{acc:.4f}", len(labels)])

        f1_table[display] = [report[c]["f1-score"] for c in CLASSES]

        with open(RESULTS / f"{run}_val_predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patch", "true_class", "predicted", "pred_confidence",
                        "true_class_prob", "correct"])
            for p, t, pr, cf, tp in zip(paths, labels, preds, confs, true_probs):
                w.writerow([p.name, CLASSES[t], CLASSES[pr], f"{cf:.4f}",
                            f"{tp:.4f}", int(t == pr)])

        cm = confusion_matrix(labels, preds, labels=range(len(CLASSES)))

        for i, actual in enumerate(CLASSES):
            for j, predicted in enumerate(CLASSES):
                if i != j and cm[i, j] > 0:
                    pair_rows.append([display, actual, predicted, int(cm[i, j]),
                                      f"{cm[i, j] / cm[i].sum() * 100:.2f}"])

        strmus = cm[CLASSES.index("STR"), CLASSES.index("MUS")]
        musstr = cm[CLASSES.index("MUS"), CLASSES.index("STR")]
        worst = max(((cm[i, j], CLASSES[i], CLASSES[j])
                     for i in range(len(CLASSES)) for j in range(len(CLASSES)) if i != j))
        print(f"  STR->MUS: {strmus}   MUS->STR: {musstr}   "
              f"largest confusion: {worst[1]}->{worst[2]} ({worst[0]})\n")

    pair_rows.sort(key=lambda r: -r[3])
    with open(RESULTS / "confusion_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "actual", "predicted", "count", "pct_of_actual"])
        w.writerows(pair_rows)

    with open(RESULTS / "all_models_per_class_f1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + CLASSES)
        for display, f1s in f1_table.items():
            w.writerow([display] + [f"{v:.4f}" for v in f1s])

    print(f"Saved per-run metrics and predictions to {RESULTS}")
    print("Run plot_eval_figures.py to redraw the confusion matrices and F1 chart.")


if __name__ == "__main__":
    main()
