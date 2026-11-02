"""
train.py — QPKI-SmartHome Training Script

Trains the DNN ThreatClassifier on CIC-IoT-2023 network flow data.
The trained model is deployed on the Raspberry Pi 4 edge gateway for
real-time threat-level scoring used by the energy-aware scheduler.

Section 4.3 / Section 6.1 of the paper.

Usage:
    python train.py
    python train.py --sample-frac 0.1   # quick test with 10% data
"""

import os
import argparse
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.ensemble import RandomForestClassifier

from config import (
    DNN, CLASSIFIER, BEST_CHECKPOINT, LAST_CHECKPOINT, RF_CHECKPOINT, LOG_DIR
)
from dataset import prepare_dataset, make_loaders
from model import ThreatClassifier, FocalLoss


# =============================================================================
# Section 6.1 — Training Configuration (matches paper Table 2)
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train QPKI-SmartHome ThreatClassifier")
    parser.add_argument("--epochs",      type=int,   default=DNN["epochs"],       help="Number of epochs")
    parser.add_argument("--batch-size",  type=int,   default=DNN["batch_size"],   help="Batch size")
    parser.add_argument("--lr",          type=float, default=DNN["learning_rate"],help="Learning rate")
    parser.add_argument("--sample-frac", type=float, default=1.0,                 help="Dataset fraction")
    parser.add_argument("--download",    action="store_true",                     help="Download dataset")
    parser.add_argument("--device",      type=str,   default="auto",              help="cpu / cuda / auto")
    parser.add_argument("--rf",          action="store_true",                     help="Also train Random Forest")
    return parser.parse_args()


def get_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, scaler_amp=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler_amp is not None:
            with torch.cuda.amp.autocast():
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds  = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total  += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds  = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total  += len(y_batch)
    return total_loss / total, correct / total


# =============================================================================
# Random Forest Baseline (Section 6.1)
# =============================================================================

def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest classifier.
    Section 6.1: RF is the primary classifier; DNN is secondary for comparison.
    """
    print("\n[train] Fitting Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=CLASSIFIER["n_estimators"],
        max_depth=CLASSIFIER["max_depth"],
        min_samples_split=CLASSIFIER["min_samples_split"],
        min_samples_leaf=CLASSIFIER["min_samples_leaf"],
        n_jobs=CLASSIFIER["n_jobs"],
        random_state=CLASSIFIER["random_state"],
    )
    rf.fit(X_train, y_train)
    val_acc = rf.score(X_val, y_val)
    print(f"[train] RF val accuracy: {val_acc:.4f}")

    with open(RF_CHECKPOINT, "wb") as f:
        pickle.dump(rf, f)
    print(f"[train] RF saved → {RF_CHECKPOINT}")
    return rf


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"[train] Device: {device}")

    # --- Data ---
    print("[train] Loading dataset …")
    data = prepare_dataset(download=args.download, sample_frac=args.sample_frac)
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    n_classes = int(np.max(y_tr)) + 1
    input_dim = X_tr.shape[1]
    print(f"[train] n_classes={n_classes}  input_dim={input_dim}")

    train_loader, val_loader, _ = make_loaders(
        X_tr, X_val, data["X_test"],
        y_tr, y_val, data["y_test"],
        batch_size=args.batch_size,
    )

    # --- Model ---
    model = ThreatClassifier(
        input_dim=input_dim,
        hidden_dims=DNN["hidden_dims"],
        n_classes=n_classes,
        dropout=DNN["dropout"],
    ).to(device)
    print(f"[train] Parameters: {model.parameter_count():,}")

    # --- Loss / Optimizer / Scheduler ---
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=DNN["weight_decay"],
    )
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision (CUDA only)
    amp_scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # TensorBoard
    writer = SummaryWriter(log_dir=LOG_DIR)

    # --- Training Loop ---
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n[train] Starting training for {args.epochs} epochs …")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        lr_scheduler.step()
        elapsed = time.time() - t0

        writer.add_scalar("Loss/train", tr_loss,  epoch)
        writer.add_scalar("Loss/val",   val_loss,  epoch)
        writer.add_scalar("Acc/train",  tr_acc,    epoch)
        writer.add_scalar("Acc/val",    val_acc,   epoch)
        writer.add_scalar("LR",         lr_scheduler.get_last_lr()[0], epoch)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={tr_loss:.4f}  acc={tr_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={lr_scheduler.get_last_lr()[0]:.2e}  "
            f"({elapsed:.1f}s)"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "input_dim": input_dim,
                "n_classes": n_classes,
                "feat_idx": data["feat_idx"],
            }, BEST_CHECKPOINT)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= DNN["early_stopping_patience"]:
            print(f"[train] Early stopping at epoch {epoch}.")
            break

    # Save last checkpoint
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_acc": val_acc,
        "input_dim": input_dim,
        "n_classes": n_classes,
        "feat_idx": data["feat_idx"],
    }, LAST_CHECKPOINT)
    print(f"[train] Last checkpoint saved → {LAST_CHECKPOINT}")
    print(f"[train] Best val accuracy: {best_val_acc:.4f}")

    # --- Optional: Random Forest ---
    if args.rf:
        train_random_forest(X_tr, y_tr, X_val, y_val)

    writer.close()
    print("[train] Done.")


if __name__ == "__main__":
    main()
