import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))  # ensure src/ is on the path

import argparse
import csv
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloaders import get_dataloaders
from models.model_factory import get_model


# -----------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# -----------------------------------------------------------------------
# One epoch of training
# -----------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# -----------------------------------------------------------------------
# One epoch of validation
# -----------------------------------------------------------------------

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# -----------------------------------------------------------------------
# Full training loop
# -----------------------------------------------------------------------

def train(
    model_name: str = "resnet18",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 5e-4,
    data_dir: str = "data",
    output_dir: str = "outputs",
    optimizer_name: str = "adamw",
    label_smoothing: float = 0.1,
    dropout: float = 0.5,
    run_name: str | None = None,
    amp: bool = False,
):
    run_name = run_name or model_name
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints" / run_name
    results_dir = output_dir / "results"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if amp and DEVICE.type != "cuda":
        raise ValueError("--amp requires CUDA, but no GPU is available")

    print(
        f"Run config: model={model_name} run_name={run_name} "
        f"optim={optimizer_name} lr={lr} wd={weight_decay} "
        f"label_smoothing={label_smoothing} dropout={dropout} "
        f"epochs={epochs} batch_size={batch_size} amp={amp}"
    )

    # --- Data ---
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
    )

    # --- Model ---
    model = get_model(model_name, pretrained=True, dropout=dropout).to(DEVICE)

    # --- Loss, optimiser, scheduler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}' (expected 'adam' or 'adamw')")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if amp else None

    # --- Logging ---
    writer = SummaryWriter(log_dir=str(output_dir / "runs" / run_name))
    csv_path = results_dir / f"{run_name}_log.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    best_val_acc = 0.0

    # --- Epoch loop ---
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        # Console
        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{elapsed:.0f}s"
        )

        # TensorBoard
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss, val_acc, current_lr])

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = checkpoints_dir / "best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  -> New best saved ({val_acc:.4f})")

    writer.close()
    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    print(f"Logs saved to {csv_path}")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           type=str,   default="resnet18")
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--weight_decay",    type=float, default=5e-4)
    parser.add_argument("--optimizer",       type=str,   default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--dropout",         type=float, default=0.5)
    parser.add_argument("--run_name",        type=str,   default=None,
                        help="Subdir under outputs/checkpoints and prefix for logs. Defaults to --model.")
    parser.add_argument("--amp",             action="store_true",
                        help="Enable mixed-precision training (CUDA only). Validation stays FP32.")
    parser.add_argument("--data_dir",        type=str,   default="../data")
    parser.add_argument("--output_dir",      type=str,   default="../outputs")
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        optimizer_name=args.optimizer,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        run_name=args.run_name,
        amp=args.amp,
    )
