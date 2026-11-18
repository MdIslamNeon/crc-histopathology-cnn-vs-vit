# CNN vs. Vision Transformer for Colorectal Histopathology

A comparative study of five image classification models on colorectal cancer tissue
patches, with Grad-CAM used to compare what each model actually looks at.

Graduate thesis project, Computer Science Department, California State University,
Los Angeles.

- Author: Md Islam
- Advisor: Dr. Navid Amini

---

## What this project asks

Pathologists look at tissue under a microscope and name what they see. This project
trains computers to do the same labeling job on small image tiles of colorectal
tissue, and then asks a question that accuracy alone cannot answer.

The question is this. Four of the models are CNNs and one is a Vision Transformer.
These are two different designs. If they all reach about the same score, are they
solving the problem the same way, or are they just arriving at the same answer by
different routes?

To find out, the project uses Grad-CAM. Grad-CAM makes a heatmap showing which parts
of an image pushed the model toward its answer. Red means "this region mattered a
lot." Blue means "this region barely mattered." Putting the heatmaps side by side
shows whether two models that agree on the label also agree on the evidence.

The short version of the finding: **same accuracy, same errors, different evidence.**

---

## Results

All five models were trained under one common protocol and evaluated on the same
7,180 validation images.

| Model | Parameters | Best val accuracy | Best epoch | Final train acc | Final val acc | Train-val gap |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 4.0 M | **97.35 %** | 2 | 99.71 % | 96.64 % | +3.07 % |
| DenseNet121 | 7.0 M | 96.78 % | 3 | 99.81 % | 95.70 % | +4.12 % |
| ViT-B/16 | 85.8 M | 96.60 % | 4 | 99.81 % | 95.52 % | +4.30 % |
| ResNet50 (regularized) | 24.0 M | 96.45 % | 4 | 99.87 % | 95.11 % | +4.76 % |
| ResNet50 (baseline) | 24.0 M | 96.23 % | 1 | 99.86 % | 94.18 % | +5.68 % |
| ResNet18 | 11.2 M | 96.09 % | 7 | 99.70 % | 95.46 % | +4.25 % |

Source file: [`outputs/results/all_models_summary.csv`](outputs/results/all_models_summary.csv)

Three things to notice.

**1. The spread is tiny.** Best to worst is about 1.3 percentage points. The smallest
model, EfficientNet-B0 at 4 million parameters, scored highest. The largest model,
ViT-B/16 at 85.8 million parameters, landed in the middle. More parameters did not
buy more accuracy here. That is useful information on its own.

**2. Every model overfits.** Training accuracy is above 99.7 % for all six runs while
validation accuracy sits near 95 to 97 %. That gap means the models are partly
memorizing the training images instead of learning tissue patterns that generalize.
The dataset comes from a limited number of slides, so there is less variety than the
100,000 image count suggests. The ResNet50 baseline was the worst case at +5.68 %,
which is why a regularized version was trained as a direct comparison.

**3. Best epoch is early.** Most models peaked between epoch 1 and 4 out of 10. The
later epochs mostly made the overfitting worse. The training script saves the best
checkpoint by validation accuracy, not the last one, so this is handled correctly.

### Per-class F1 scores

F1 is a single score per tissue class, from 0 to 1, that balances "how often it is
right when it says this class" against "how many of this class it finds."

| Model | ADI | BACK | DEB | LYM | MUC | MUS | NORM | STR | TUM |
|---|---|---|---|---|---|---|---|---|---|
| ResNet18 | .977 | 1.000 | .994 | .979 | .977 | .863 | .969 | **.795** | .982 |
| ResNet50 (baseline) | .988 | .999 | .997 | .988 | .975 | .873 | .971 | **.749** | .983 |
| ResNet50 (reg) | .979 | .996 | .980 | .988 | .962 | .900 | .983 | **.834** | .975 |
| DenseNet121 | .987 | .999 | .987 | .992 | .976 | .869 | .970 | **.855** | .984 |
| EfficientNet-B0 | .993 | 1.000 | .987 | .996 | .995 | .881 | .982 | **.847** | .985 |
| ViT-B/16 | .988 | .998 | .981 | .994 | .978 | .880 | .973 | **.812** | .981 |

Source file: [`outputs/results/all_models_per_class_f1.csv`](outputs/results/all_models_per_class_f1.csv)

**This table is the heart of the project.** Seven of the nine classes score above
0.96 for every model. Two do not. STR sits between 0.75 and 0.86, and MUS sits
between 0.86 and 0.90. Every single model struggles on the same two classes.

STR is stroma, the connective tissue scaffolding. MUS is smooth muscle. Under H&E
stain both are pink, fibrous, and directional. They genuinely look alike, even to
trained eyes. The models confuse them with each other, not with the other seven
classes.

That shared weakness is what makes STR and MUS the target of the Grad-CAM analysis.
When six different architectures fail on the same boundary, the interesting question
is whether they fail for the same visual reason.

### The nine tissue classes

| Code | Tissue |
|---|---|
| ADI | Adipose (fat) |
| BACK | Background (no tissue) |
| DEB | Debris |
| LYM | Lymphocytes (immune cells) |
| MUC | Mucus |
| MUS | Smooth muscle |
| NORM | Normal colon mucosa |
| STR | Cancer-associated stroma |
| TUM | Colorectal adenocarcinoma epithelium (tumor) |

---

## What is in this repository, and what is not

This repo holds the **code and the results**. It does not hold the dataset or the
trained model weights, because both are far too large for GitHub.

### Included

| Path | Contents |
|---|---|
| `src/` | All Python. Training, evaluation, Grad-CAM, plotting. |
| `outputs/results/` | Every number, as CSV. Training logs, per-class metrics, per-image predictions, Grad-CAM metrics. |
| `outputs/figures/` | Every generated figure, as PNG. |
| `outputs/channel_mean.npy`, `channel_std.npy` | Dataset normalization statistics. **Required at runtime.** Do not delete these. |
| `requirements.txt` | Python dependencies. |

### Not included

| What | Size | How to get it |
|---|---|---|
| `data/` | ~15.5 GB, 107,180 files | Download from Zenodo. See below. |
| `outputs/checkpoints/` | ~593 MB | Retrain. See below. GitHub rejects any file over 100 MB, and the ViT checkpoint alone is 327 MB. |
| `outputs/runs/` | TensorBoard event files | Regenerated automatically on every training run. |

Because the result CSVs and figures are committed, **you can read every finding in
this project without downloading the dataset or owning a GPU.** You only need those
to reproduce or extend the work.

---

## Setup

### 1. Get the code

```bash
git clone https://github.com/MdIslamNeon/crc-histopathology-cnn-vs-vit.git
cd crc-histopathology-cnn-vs-vit
```

### 2. Make a Python environment

Python 3.10 or newer. A virtual environment keeps these packages from colliding with
anything else on your machine.

```bash
python -m venv .venv
```

Activate it. On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Windows Git Bash:

```bash
source .venv/Scripts/activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install PyTorch

Install PyTorch **first and separately**, because the right command depends on your
GPU and CUDA version. Get the exact command from
<https://pytorch.org/get-started/locally/>.

A CPU-only install will run the plotting scripts fine, but training will be
impractically slow. You want a CUDA build and an NVIDIA GPU for anything involving
training or Grad-CAM.

### 4. Install everything else

```bash
pip install -r requirements.txt
```

### 5. Download the dataset

The dataset is **NCT-CRC-HE-100K**, published by Kather et al. It is free and public.

Download page: <https://zenodo.org/records/1214456>

Two archives are needed:

| Archive | Use here | Images |
|---|---|---|
| `NCT-CRC-HE-100K.zip` | training set | 100,000 |
| `CRC-VAL-HE-7K.zip` | validation set | 7,180 |

Unzip them so the folder layout looks exactly like this:

```
crc-histopathology-cnn-vs-vit/
├── data/
│   ├── train/            <- from NCT-CRC-HE-100K.zip
│   │   ├── ADI/          *.tif
│   │   ├── BACK/
│   │   ├── DEB/
│   │   ├── LYM/
│   │   ├── MUC/
│   │   ├── MUS/
│   │   ├── NORM/
│   │   ├── STR/
│   │   └── TUM/
│   └── val/              <- from CRC-VAL-HE-7K.zip
│       ├── ADI/
│       └── ... same nine class folders
├── src/
└── outputs/
```

The class folder names must match exactly, in capitals. The loader reads the folder
name to get the label, so a renamed folder silently becomes a different class.

Images are 224x224 pixel RGB `.tif` tiles of H&E stained tissue, at 0.5 microns per
pixel. They are already the right size for every model here, so no resizing is needed.

Expected image counts per class:

| Class | Train | Val |
|---|---|---|
| ADI | 10,407 | 1,338 |
| BACK | 10,566 | 847 |
| DEB | 11,512 | 339 |
| LYM | 11,557 | 634 |
| MUC | 8,896 | 1,035 |
| MUS | 13,536 | 592 |
| NORM | 8,763 | 741 |
| STR | 10,446 | 421 |
| TUM | 14,317 | 1,233 |
| **Total** | **100,000** | **7,180** |

Note that train and validation come from **different patient cohorts**. That is the
right way to split this data, and it is why validation accuracy is a fair estimate
rather than an inflated one.

### 6. Check that it worked

```bash
cd src
python eda.py
```

Run this one from the project root instead if it complains about paths:

```bash
python src/eda.py
```

It prints the class counts and writes sample images to `outputs/`. If the counts
match the table above, the dataset is in place correctly.

---

## Reproducing the results

Run the stages in this order. Each stage reads what the previous one wrote.

Unless noted, run the scripts from inside `src/`:

```bash
cd src
```

Most scripts find the project folders from their own file location, so they work from
anywhere. The two exceptions are `train.py`, whose `--data_dir` default of `../data`
assumes you are in `src/`, and `eda.py`, which expects to be run from the project
root.

### Stage 1: Train the models

Six runs. These are the exact commands behind the results table.

```bash
# Four models under the common protocol
python train.py --model resnet18        --run_name resnet18
python train.py --model densenet121     --run_name densenet121
python train.py --model efficientnet_b0 --run_name efficientnet_b0

# Vision Transformer. --amp enables mixed precision to fit in VRAM.
python train.py --model vit_b16 --run_name vit_b16 --amp

# ResNet50, regularized. Same protocol as the others.
python train.py --model resnet50 --run_name resnet50_reg

# ResNet50, deliberately under-regularized, to show the overfitting contrast.
python train.py --model resnet50 --run_name resnet50 \
    --optimizer adam --weight_decay 1e-4 --dropout 0.0 --label_smoothing 0.0
```

**The common training protocol** is the set of defaults in `train.py`:

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4, cosine decay to 0 |
| Weight decay | 5e-4 |
| Label smoothing | 0.1 |
| Dropout before the classifier | 0.5 |
| Batch size | 32 |
| Epochs | 10 |
| Pretrained weights | ImageNet, for every model |
| Precision | FP32, except ViT which uses AMP |

Two honest caveats about calling this a controlled comparison.

The protocol is **common, not identical**. The models have different parameter
counts and different internal structures, so "identical settings" is not achievable
even in principle. ViT also uses mixed precision while the CNNs use FP32, which was a
VRAM necessity rather than a free choice. What is held constant is the data, the
split, the augmentation, the optimizer, the schedule, the epoch budget, and the
checkpoint selection rule.

Also, the hyperparameters were **not tuned per model**. They were chosen once and
applied to all. A model that performs worse here might perform better with its own
tuned settings. The comparison answers "how do these architectures behave under one
reasonable protocol," not "what is each architecture's ceiling."

Expect roughly 10 minutes per epoch for ResNet50 on a single consumer GPU, so about
1.5 to 2 hours per run. Smaller models are faster, ViT is slower. Budget most of a
day for all six.

Each run writes:

- `outputs/checkpoints/{run_name}/best.pth` - best epoch by validation accuracy
- `outputs/results/{run_name}_log.csv` - per-epoch loss, accuracy, learning rate
- `outputs/runs/{run_name}/` - TensorBoard event files

Watch training live with:

```bash
tensorboard --logdir ../outputs/runs
```

### Stage 2: Evaluate

```bash
python evaluate.py
```

Runs every checkpoint over the full validation set. Writes per-class metrics and, for
each model, a per-image prediction CSV. Those prediction files are what the Grad-CAM
stage uses to find the images the models got wrong.

Outputs: `outputs/results/{run}_per_class_metrics.csv`,
`{run}_val_predictions.csv`, `all_models_per_class_f1.csv`, `confusion_pairs.csv`

### Stage 3: Grad-CAM

```bash
python gradcam_explainer.py    # teaching figure: how to read a Grad-CAM heatmap
python gradcam_compare.py      # all five models on the same STR and MUS patches
python gradcam_confusion.py    # heatmaps for the images the models got wrong
python gradcam_metrics.py      # turns the heatmaps into numbers
```

`gradcam_compare.py` is the central deliverable. It puts five architectures on one
row for the same input patch, so the differences in evidence are visible directly.

`gradcam_metrics.py` is what keeps this from being a "look at the pretty pictures"
argument. It quantifies the heatmaps three ways:

- **Properties** - how spread out or concentrated each heatmap is
- **Faithfulness** - mask the regions a heatmap called important, then check how far
  accuracy drops. A big drop means the heatmap was pointing at something real.
- **Agreement** - how similar two models' heatmaps are on the same image. This is the
  measurement behind the "different evidence" claim.

### Stage 4: Figures

```bash
python plot_eval_figures.py             # confusion matrices, per-class F1
python plot_all_models_comparison.py    # accuracy comparison, accuracy vs parameters
python plot_resnet50_comparison.py      # overfitting: baseline vs regularized
python plot_str_mus_examples.py         # example STR and MUS patches
python plot_cam_properties.py           # Grad-CAM property charts
```

These read the CSVs rather than the checkpoints, so restyling a figure does not need
a GPU or a re-run.

---

## Where the Grad-CAM work is handled

A Vision Transformer does not have convolutional feature maps, so Grad-CAM needs an
extra step for it. ViT processes the image as a sequence of 196 patch tokens plus one
class token. The code drops the class token, because it has no position in the image,
and folds the remaining 196 tokens back into a 14x14 grid that Grad-CAM can treat
like a CNN feature map. See `vit_reshape` in `src/gradcam_compare.py`.

Target layers per architecture, also in `gradcam_compare.py`:

| Architecture | Grad-CAM target layer |
|---|---|
| ResNet18 / ResNet50 | `model.layer4[-1]` |
| DenseNet121 | `model.features[-1]` |
| EfficientNet-B0 | `model.features[-1]` |
| ViT-B/16 | last encoder block, reshaped to 14x14 |

Getting ViT and full Grad-CAM working together is the methodological contribution
here. Papers on this dataset tend to do one or the other.

---

## File guide

### Code

| File | What it does |
|---|---|
| `src/dataset.py` | `CRCDataset`, the class list, normalization constants. Supports `fraction=` for reduced-data experiments. |
| `src/dataloaders.py` | Training augmentation, validation transforms, batching. |
| `src/models/model_factory.py` | `get_model(name)`. The architecture registry. |
| `src/models/baseline.py` | The original single-model baseline, kept for reference. |
| `src/train.py` | Training loop, CSV and TensorBoard logging, best-checkpoint saving. |
| `src/evaluate.py` | Full validation pass for every run. |
| `src/eda.py` | Class counts, sample images, channel mean and std. |
| `src/tissue_classes.py` | Tissue class reference figure. |
| `src/gradcam_*.py` | Grad-CAM generation, confusion analysis, and metrics. |
| `src/plot_*.py` | Figure generation from the result CSVs. |

`model_factory.py` also registers `resnet101`, `efficientnet_b3`, and `mobilenet_v3`.
Those were considered and then cut from the thesis scope. They should still work if
you want them.

### Key result files

| File | What is in it |
|---|---|
| `all_models_summary.csv` | The headline accuracy table |
| `all_models_per_class_f1.csv` | Per-class F1 for all six runs |
| `confusion_pairs.csv` | Which classes get mistaken for which |
| `{run}_log.csv` | Per-epoch training history |
| `{run}_val_predictions.csv` | Per-image predictions. Useful for any new analysis. |
| `cam_agreement.csv` | Do two models highlight the same regions |
| `cam_faithfulness.csv` | Do the highlighted regions actually matter |
| `cam_properties.csv` | Heatmap spread and concentration |
| `consensus_hard_cases.csv` | Images that every model got wrong |
| `strmus_dissent.csv` | Images where the models disagreed with each other |

### Key figures

| File | What it shows |
|---|---|
| `gradcam_STR.png`, `gradcam_MUS.png` | The central comparison. Five models, same patches. |
| `gradcam_contrastive_STR_vs_MUS.png` | STR evidence against MUS evidence |
| `gradcam_confusion_*.png` | Heatmaps on misclassified images |
| `gradcam_explained.png` | How to read a Grad-CAM heatmap |
| `cam_agreement_matrix.png` | Model-to-model heatmap similarity |
| `cam_faithfulness_curves.png` | Accuracy drop as highlighted regions are masked |
| `all_models_val_accuracy.png` | Accuracy comparison |
| `accuracy_vs_params.png` | Accuracy against model size |
| `resnet50_baseline_vs_regularized.png` | The overfitting contrast |
| `{run}_confusion_matrix.png` | Per-model confusion matrix |
| `str_mus_examples.png` | Why STR and MUS are hard to tell apart |
| `tissue_classes.png` | One example per class |

---

## Things to know before you change anything

**`outputs/channel_mean.npy` and `channel_std.npy` are required.** `dataset.py` loads
them to normalize images. If they are missing it silently falls back to ImageNet
statistics, which will shift every result slightly and make your numbers not match
this table. Both files are committed here. Leave them alone. If you ever regenerate
them with `eda.py`, every model has to be retrained to stay comparable.

**Run names are hardcoded in the analysis scripts.** `evaluate.py` and
`gradcam_compare.py` each hold a `RUNS` list near the top. If you train a model under
a new `--run_name`, add it to those lists or the analysis will skip it.

**`--run_name` controls the output paths**, not `--model`. Two runs of the same
architecture need two different run names, or the second will overwrite the first.
That is how `resnet50` and `resnet50_reg` coexist.

**Checkpoints store a dropout-aware classifier head.** `evaluate.py` tries dropout
0.5 and then 0.0 when loading, so older checkpoints from before dropout was added
still load. Keep that behavior if you touch the loading code.

**The accuracy numbers will not match to the last decimal.** Data loader shuffling and
GPU nondeterminism mean a retrain lands within a few tenths of a percent, not exactly
on these values. If you land within about half a percentage point, your run is
healthy. If you are several points off, something is wrong with the data layout or
the normalization statistics.

---

## Ideas for whoever picks this up

Ordered roughly by effort.

**Close the overfitting gap.** Every model sits near a 3 to 5 point train-validation
gap. Stronger augmentation, early stopping, or fewer epochs with a tuned schedule are
all unexplored here. The `fraction=` argument in `CRCDataset` makes it cheap to test
how much data is actually needed.

**Attack STR versus MUS directly.** This is the one real weakness in the results. A
two-class classifier trained only on STR and MUS, or a loss weighted toward that
boundary, would say whether the confusion is a capacity problem or a genuine
ambiguity in the images.

**Stain normalization.** Slides stained on different days come out different colors,
and the models may be leaning on color more than structure. This was scoped out of
the thesis and never tested. Macenko or Reinhard normalization before training would
answer it. `requirements.txt` no longer pins `staintools`, so add it if you go this
way.

**Tune per model.** The current protocol is deliberately uniform. Giving each
architecture its own tuned hyperparameters would answer the different and also
interesting question of what each one can actually reach.

**Validate on a different cohort.** Everything here is NCT-CRC-HE-100K. Testing the
trained models on another colorectal dataset would show whether the findings hold
beyond this one source.

**Whole-slide context.** These are isolated 224x224 tiles. A pathologist sees
surrounding tissue. Any method that reintroduces spatial context is a substantial but
natural extension.

**Architectures that were cut.** `mobilenet_v3`, `resnet101`, and `efficientnet_b3`
are already registered in `model_factory.py` and were dropped for scope, not for any
technical reason. A hybrid CNN-Transformer, self-supervised pretraining, and knowledge
distillation were also considered and cut.

---

## Citation

If you use this code, please cite the dataset authors. The dataset is the substantial
contribution here.

```bibtex
@dataset{kather_2018_1214456,
  author    = {Kather, Jakob Nikolas and Halama, Niels and Marx, Alexander},
  title     = {{100,000 histological images of human colorectal
                cancer and healthy tissue}},
  month     = apr,
  year      = 2018,
  publisher = {Zenodo},
  version   = {v0.1},
  doi       = {10.5281/zenodo.1214456},
  url       = {https://doi.org/10.5281/zenodo.1214456}
}
```

Grad-CAM comes from Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization," ICCV 2017. The implementation used here is
the `pytorch-grad-cam` package by Jacob Gildenblat.

---

## A note on the dataset terms

NCT-CRC-HE-100K is distributed under CC BY 4.0 by its authors. Check the Zenodo
record for current terms before redistributing any of it. This repository does not
contain dataset images.
