"""
==============================================================
  PLANT DISEASE DETECTION — PHASE 3 & 4: Training Pipeline
==============================================================
  What this file does:
    1. Loads the dataset splits created in Phase 1
    2. Initializes the CNN model from Phase 2
    3. Configures optimizer, loss function, LR scheduler
    4. Trains for N epochs with validation after each epoch
    5. Saves the best model checkpoint automatically
    6. Plots training/validation loss and accuracy curves
    7. Implements early stopping to prevent overfitting
==============================================================
"""

import os
import time
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import our model and data prep from previous phases
from model import build_model
from data_prep import get_dataloaders, SPLIT_OUTPUT_PATH


# ==============================================================
# CONFIGURATION — Tune these hyperparameters
# ==============================================================

NUM_CLASSES   = 38          # PlantVillage has 38 classes
NUM_EPOCHS    = 30          # Max training epochs
LEARNING_RATE = 0.001       # Initial learning rate for Adam
WEIGHT_DECAY  = 1e-4        # L2 regularization (prevents overfitting)
PATIENCE      = 7           # Early stopping: stop after N epochs no improvement
SAVE_PATH     = "best_model.pth"   # Where to save the best weights
HISTORY_PATH  = "training_history.json"


# ==============================================================
# STEP 1 — Setup: device, model, loss, optimizer, scheduler
# ==============================================================

def setup_training():
    """
    Initializes everything needed before training starts.
    Returns: model, criterion, optimizer, scheduler, device
    """

    # Auto-detect GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device     : {device.upper()}")
    if device == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Build model
    model = build_model(num_classes=NUM_CLASSES, device=device)

    # Loss function
    # CrossEntropyLoss = Softmax + NLLLoss combined
    # Perfect for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Optimizer: Adam
    # Adam adapts learning rates per parameter — generally best for CNNs
    # weight_decay adds L2 penalty to weights to reduce overfitting
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # LR Scheduler: ReduceLROnPlateau
    # Automatically reduces LR when validation loss stops improving
    # factor=0.5 → LR halved when plateau detected
    # patience=3 → waits 3 epochs before reducing
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',       # Monitor minimum (val loss)
        factor=0.5,
        patience=3,
        min_lr=1e-6       # Never go below this LR
    )

    print(f"  Model      : PlantDiseaseNet ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"  Optimizer  : Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"  Scheduler  : ReduceLROnPlateau (factor=0.5, patience=3)")
    print(f"  Loss       : CrossEntropyLoss")
    print(f"  Epochs     : {NUM_EPOCHS} (early stop patience={PATIENCE})")

    return model, criterion, optimizer, scheduler, device


# ==============================================================
# STEP 2 — Train for one epoch
# ==============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs the full training loop for one epoch.

    For each batch:
      1. Forward pass  → get predictions
      2. Compute loss  → compare predictions to true labels
      3. Backward pass → compute gradients
      4. Optimizer step → update weights

    Returns: average loss and accuracy for this epoch
    """
    model.train()   # Set model to training mode (enables dropout, batchnorm updates)

    running_loss     = 0.0
    correct          = 0
    total            = 0
    num_batches      = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader):

        # Move data to GPU/CPU
        images = images.to(device)
        labels = labels.to(device)

        # --- Forward pass ---
        optimizer.zero_grad()           # Clear gradients from last step
        outputs = model(images)         # Get raw predictions (logits)
        loss = criterion(outputs, labels)  # Compute loss

        # --- Backward pass ---
        loss.backward()                 # Compute gradients
        optimizer.step()                # Update model weights

        # --- Track metrics ---
        running_loss += loss.item()
        _, predicted = outputs.max(1)   # Get class with highest score
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            current_acc = 100.0 * correct / total
            print(f"    Batch [{batch_idx+1:>4}/{num_batches}]  "
                  f"Loss: {running_loss/(batch_idx+1):.4f}  "
                  f"Acc: {current_acc:.1f}%")

    epoch_loss = running_loss / num_batches
    epoch_acc  = 100.0 * correct / total
    return epoch_loss, epoch_acc


# ==============================================================
# STEP 3 — Validate (no gradient updates)
# ==============================================================

def validate(model, dataloader, criterion, device):
    """
    Runs the validation loop — no weight updates, just measurement.

    model.eval() turns off:
      - Dropout (use all neurons for evaluation)
      - BatchNorm update (use running statistics)

    torch.no_grad() disables gradient computation → faster & less memory
    """
    model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc  = 100.0 * correct / total
    return epoch_loss, epoch_acc


# ==============================================================
# STEP 4 — Early stopping handler
# ==============================================================

class EarlyStopping:
    """
    Stops training when validation loss stops improving.

    Why early stopping?
      Without it, the model keeps training and starts memorizing the
      training data instead of learning generalizable patterns
      (overfitting). Early stopping halts when we detect this.
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float('inf')
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            print(f"    [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ==============================================================
# STEP 5 — Full training loop
# ==============================================================

def train(model, dataloaders, criterion, optimizer, scheduler, device):
    """
    Main training loop — runs for NUM_EPOCHS or until early stopping.

    Each epoch:
      1. Train on training set
      2. Evaluate on validation set
      3. Step LR scheduler based on val loss
      4. Save model if val loss improved
      5. Check early stopping

    Returns: training history dict with losses and accuracies
    """

    early_stopping = EarlyStopping(patience=PATIENCE)
    best_val_loss  = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr":         []
    }

    print("\n" + "="*60)
    print("  TRAINING STARTED")
    print("="*60)

    total_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):

        epoch_start = time.time()
        current_lr  = optimizer.param_groups[0]['lr']

        print(f"\n  Epoch [{epoch:>2}/{NUM_EPOCHS}]  LR: {current_lr:.6f}")
        print("  " + "-"*50)

        # --- Train ---
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device
        )

        # --- Validate ---
        val_loss, val_acc = validate(
            model, dataloaders["val"], criterion, device
        )

        # --- LR Scheduler step ---
        scheduler.step(val_loss)

        # --- Save history ---
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        # --- Print epoch summary ---
        print(f"\n  Train  →  Loss: {train_loss:.4f}  |  Acc: {train_acc:.2f}%")
        print(f"  Val    →  Loss: {val_loss:.4f}  |  Acc: {val_acc:.2f}%")
        print(f"  Time   →  {epoch_time:.1f}s")

        # --- Save best model ---
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  [✓] New best model saved  (val_loss={val_loss:.4f})")

        # --- Early stopping check ---
        if early_stopping(val_loss):
            print(f"\n  [!] Early stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - total_start
    print(f"\n  Training complete in {total_time/60:.1f} minutes")
    print(f"  Best val loss : {best_val_loss:.4f}")

    # Load best weights back into model
    model.load_state_dict(best_model_wts)

    # Save training history to JSON
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [✓] History saved to '{HISTORY_PATH}'")

    return model, history


# ==============================================================
# STEP 6 — Plot training curves
# ==============================================================

def plot_training_curves(history: dict):
    """
    Plots loss and accuracy curves for training and validation.
    These help you diagnose:
      - Overfitting: train acc >> val acc
      - Underfitting: both accuracies are low
      - Good fit: both curves converge and are close together
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — PlantDiseaseNet", fontsize=14)

    # --- Loss curve ---
    ax1.plot(epochs, history["train_loss"], 'b-o', label="Train loss", markersize=4)
    ax1.plot(epochs, history["val_loss"],   'r-o', label="Val loss",   markersize=4)
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- Accuracy curve ---
    ax2.plot(epochs, history["train_acc"], 'b-o', label="Train acc", markersize=4)
    ax2.plot(epochs, history["val_acc"],   'r-o', label="Val acc",   markersize=4)
    ax2.set_title("Accuracy over epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  [✓] Saved training curves to 'training_curves.png'")


# ==============================================================
# MAIN — Run the full training pipeline
# ==============================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  PLANT DISEASE DETECTION — PHASE 3 & 4: Training")
    print("="*60)

    # 1. Load data
    print("\n  Loading dataloaders...")
    dataloaders, class_names = get_dataloaders(SPLIT_OUTPUT_PATH)

    # 2. Setup model, optimizer, scheduler
    model, criterion, optimizer, scheduler, device = setup_training()

    # 3. Train
    model, history = train(
        model, dataloaders, criterion, optimizer, scheduler, device
    )

    # 4. Plot curves
    plot_training_curves(history)

    print("\n" + "="*60)
    print("  PHASE 3 & 4 COMPLETE!")
    print(f"  Best model saved to : '{SAVE_PATH}'")
    print(f"  History saved to    : '{HISTORY_PATH}'")
    print("  Next step: Phase 5 → Evaluation")
    print("="*60 + "\n")