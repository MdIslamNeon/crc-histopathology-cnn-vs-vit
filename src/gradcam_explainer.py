"""
Teaching figure: the five steps Grad-CAM takes, using a real patch and real
activations from a trained model rather than a schematic.

Writes:
    outputs/figures/gradcam_explained.png
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import CLASSES
from gradcam_compare import VAL, FIGURES, load_batch, load_run

warnings.filterwarnings("ignore")

RUN, ARCH, DISPLAY = "resnet50_reg", "resnet50", "ResNet50"
PATCH = VAL / "STR" / "STR-TCGA-AAMALCER.tif"
N_MAPS = 4


def grad_cam_pieces(model, batch):
    """Run Grad-CAM by hand so the intermediate tensors are available."""
    store = {}
    layer = model.layer4[-1]
    h1 = layer.register_forward_hook(lambda m, i, o: store.update(acts=o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.update(grads=go[0]))

    logits = model(batch)
    probs = torch.softmax(logits, dim=1)[0]
    cls = int(logits.argmax(1))
    model.zero_grad()
    logits[0, cls].backward()

    h1.remove()
    h2.remove()

    acts = store["acts"][0].detach()                 # (C, h, w)
    weights = store["grads"][0].detach().mean(dim=(1, 2))   # (C,)
    cam = torch.relu((weights[:, None, None] * acts).sum(0))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return acts.cpu(), weights.cpu(), cam.cpu().numpy(), cls, float(probs[cls])


def main():
    rgb, batch = load_batch([PATCH])
    model = load_run(RUN, ARCH)
    acts, weights, cam, cls, prob = grad_cam_pieces(model, batch)

    n_ch, gh, gw = acts.shape
    # Rank by actual positive contribution, not weight alone - a big weight on a
    # map that never fires produces an empty, unilluminating thumbnail.
    contrib = (weights[:, None, None] * acts).clamp(min=0).sum(dim=(1, 2))
    top = torch.argsort(contrib, descending=True)[:N_MAPS]

    fig = plt.figure(figsize=(19, 4.6))
    gs = fig.add_gridspec(1, 9, width_ratios=[1, .22, 1, .22, 1, .22, 1, .22, 1],
                          wspace=0.32)
    # Panel titles are kept to two short lines each. Anything longer than the
    # panel is wide spills sideways into the neighbouring title, because
    # Matplotlib does not clip or wrap set_title text.
    TITLE_KW = dict(fontsize=11, pad=8)

    def arrow(col, x=0.5):
        ax = fig.add_subplot(gs[0, col])
        ax.axis("off")
        ax.text(x, 0.55, "→", ha="center", va="center", fontsize=30, color="0.4")

    # 1. input
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb[0])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("1. Input patch\ntrue class: STR", **TITLE_KW)
    arrow(1)

    # 2. feature maps from the last conv layer
    holder = fig.add_subplot(gs[0, 2])
    holder.axis("off")
    holder.set_title(f"2. Last conv layer\n{n_ch} feature maps, {gh}x{gw} each",
                     **TITLE_KW)
    inner = gs[0, 2].subgridspec(2, 2, wspace=0.06, hspace=0.22)
    for k, ch in enumerate(top):
        a = fig.add_subplot(inner[k // 2, k % 2])
        a.imshow(acts[ch], cmap="viridis")
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"map #{int(ch)}", fontsize=8, pad=2)
    arrow(3)

    # 3. gradient-derived importance weight per map
    ax = fig.add_subplot(gs[0, 4])
    order = torch.argsort(weights, descending=True)
    show = torch.cat([order[:8], order[-8:]])
    vals = weights[show].numpy() * 1e4
    ax.bar(range(len(vals)), vals,
           color=["#1e8449" if v > 0 else "#c0392b" for v in vals])
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xticks([])
    # The unit lives in the title rather than a y-axis label: a ylabel reaches
    # left out of the panel and collides with the incoming arrow.
    ax.set_title("3. Gradient weight per map " r"($\times 10^{-4}$)"
                 "\n8 most positive, 8 most negative", **TITLE_KW)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    arrow(5)

    # 4. weighted sum, ReLU, still at feature-map resolution
    ax = fig.add_subplot(gs[0, 6])
    ax.imshow(cam, cmap="jet")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"4. Weighted sum, keep positives\ncoarse {gh}x{gw} evidence map",
                 **TITLE_KW)
    arrow(7)

    # 5. upsample and overlay
    ax = fig.add_subplot(gs[0, 8])
    big = torch.nn.functional.interpolate(
        torch.from_numpy(cam)[None, None], size=(224, 224),
        mode="bilinear", align_corners=False)[0, 0].numpy()
    ax.imshow(show_cam_on_image(rgb[0], big, use_rgb=True))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"5. Upsample to 224x224\nprediction: {CLASSES[cls]}  p={prob:.2f}",
                 **TITLE_KW)

    fig.suptitle(f"Grad-CAM, step by step — {DISPLAY} on a stroma patch",
                 fontsize=14, y=1.06)
    out = FIGURES / "gradcam_explained.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
