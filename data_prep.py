"""
==============================================================
  PLANT DISEASE DETECTION — PHASE 1: Data Preparation
==============================================================
  What this file does:
    1. Scans your PlantVillage dataset folder
    2. Shows dataset statistics (class names, image counts)
    3. Splits data into Train / Validation / Test sets (80/10/10)
    4. Creates PyTorch DataLoaders with preprocessing & augmentation
    5. Visualizes sample images from the dataset
    6. Saves class names to a JSON file for later use
==============================================================
"""

import os
import json
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ==============================================================
# CONFIGURATION — Change these paths to match your setup
# ==============================================================

# Path to the 'color' folder inside your PlantVillage dataset
DATASET_PATH = "dataset/plantvillage"

# Where to save the organized train/val/test split
SPLIT_OUTPUT_PATH = "dataset/split"

# Training settings
IMAGE_SIZE = 224        # CNN input size (224x224 pixels)
BATCH_SIZE = 32         # Images processed per step
NUM_WORKERS = 2         # Parallel data loading workers
RANDOM_SEED = 42        # For reproducibility

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10


# ==============================================================
# STEP 1 — Explore and understand the dataset
# ==============================================================

def explore_dataset(dataset_path: str):
    """
    Scan the dataset folder and print statistics.
    Returns a dict: {class_name: [list of image paths]}
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {dataset_path.resolve()}\n"
            "Please download PlantVillage from Kaggle and place the 'color' folder "
            "contents inside 'dataset/plantvillage/'\n"
        )

    class_data = {}
    total_images = 0

    print("\n" + "="*60)
    print("  DATASET EXPLORATION")
    print("="*60)
    print(f"  Path  : {dataset_path.resolve()}")

    for class_folder in sorted(dataset_path.iterdir()):
        if not class_folder.is_dir():
            continue

        # Collect valid image files
        images = [
            str(f) for f in class_folder.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ]

        if images:
            class_data[class_folder.name] = images
            total_images += len(images)

    print(f"  Classes: {len(class_data)}")
    print(f"  Total images: {total_images:,}")
    print("\n  Class breakdown:")
    print(f"  {'Class Name':<45} {'Images':>8}")
    print("  " + "-"*55)

    for class_name, images in class_data.items():
        print(f"  {class_name:<45} {len(images):>8,}")

    print("="*60)
    return class_data


# ==============================================================
# STEP 2 — Split dataset into Train / Val / Test
# ==============================================================

def split_dataset(class_data: dict, output_path: str, seed: int = 42):
    """
    Splits each class into train/val/test and copies files
    into an organized folder structure:

    split/
      train/  ← 80% of each class
        Apple___Apple_scab/
        ...
      val/    ← 10% of each class
      test/   ← 10% of each class
    """
    output_path = Path(output_path)
    random.seed(seed)

    # Skip if already split
    if (output_path / "train").exists():
        print(f"\n[INFO] Split already exists at: {output_path}")
        print("       Delete the 'split' folder to re-run.\n")
        return

    print("\n" + "="*60)
    print("  SPLITTING DATASET  (80% train / 10% val / 10% test)")
    print("="*60)

    split_counts = {"train": 0, "val": 0, "test": 0}

    for class_name, images in class_data.items():
        # Shuffle for randomness
        images_shuffled = images.copy()
        random.shuffle(images_shuffled)

        total = len(images_shuffled)
        n_train = int(total * TRAIN_RATIO)
        n_val   = int(total * VAL_RATIO)
        # Test gets the remainder to avoid rounding issues
        n_test  = total - n_train - n_val

        splits = {
            "train": images_shuffled[:n_train],
            "val":   images_shuffled[n_train : n_train + n_val],
            "test":  images_shuffled[n_train + n_val:]
        }

        for split_name, file_list in splits.items():
            dest_dir = output_path / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for src_path in file_list:
                shutil.copy2(src_path, dest_dir / Path(src_path).name)

            split_counts[split_name] += len(file_list)

    print(f"  Train images : {split_counts['train']:,}")
    print(f"  Val images   : {split_counts['val']:,}")
    print(f"  Test images  : {split_counts['test']:,}")
    print(f"\n  Saved to: {output_path.resolve()}")
    print("="*60)


# ==============================================================
# STEP 3 — Define image transforms (preprocessing + augmentation)
# ==============================================================

def get_transforms():
    """
    Returns a dict with train/val/test transforms.

    TRAIN transforms include augmentation to:
      - Prevent overfitting
      - Make model robust to real photos (different angles, lighting)

    VAL/TEST transforms only resize and normalize — no augmentation
    because we want consistent evaluation metrics.
    """

    # ImageNet mean and std — standard normalization for RGB models
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),       # Flip left-right
        transforms.RandomVerticalFlip(p=0.2),         # Flip upside-down
        transforms.RandomRotation(degrees=30),         # Rotate up to 30°
        transforms.ColorJitter(                        # Random color tweaks
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.ToTensor(),                         # Convert to tensor [0,1]
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return {
        "train": train_transform,
        "val":   val_test_transform,
        "test":  val_test_transform,
    }


# ==============================================================
# STEP 4 — Create PyTorch DataLoaders
# ==============================================================

def get_dataloaders(split_path: str):
    """
    Loads the split dataset and returns DataLoaders ready for training.
    DataLoaders handle batching, shuffling, and parallel loading.
    """
    split_path = Path(split_path)
    transforms_dict = get_transforms()

    datasets_dict = {
        split: datasets.ImageFolder(
            root=str(split_path / split),
            transform=transforms_dict[split]
        )
        for split in ["train", "val", "test"]
    }

    dataloaders = {
        "train": DataLoader(
            datasets_dict["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,           # Shuffle every epoch during training
            num_workers=NUM_WORKERS,
            pin_memory=True,        # Faster GPU transfer
        ),
        "val": DataLoader(
            datasets_dict["val"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets_dict["test"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ),
    }

    # Print summary
    print("\n" + "="*60)
    print("  DATALOADERS READY")
    print("="*60)
    for split, ds in datasets_dict.items():
        print(f"  {split:<8}: {len(ds):>6,} images  |  "
              f"{len(dataloaders[split]):>4} batches  |  "
              f"{len(ds.classes)} classes")
    print("="*60)

    return dataloaders, datasets_dict["train"].classes


# ==============================================================
# STEP 5 — Save class names to JSON (needed during inference)
# ==============================================================

def save_class_names(class_names: list, output_file: str = "class_names.json"):
    """
    Saves the list of class names to a JSON file.
    This is important — during inference you need to convert
    model output index → disease name.
    """
    with open(output_file, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\n  [✓] Saved {len(class_names)} class names to '{output_file}'")


# ==============================================================
# STEP 6 — Visualize sample images
# ==============================================================

def visualize_samples(dataloaders: dict, class_names: list, n_samples: int = 16):
    """
    Shows a grid of sample training images with their class labels.
    This helps you verify the dataset loaded correctly.
    """
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

    # Get one batch
    images, labels = next(iter(dataloaders["train"]))
    images = images[:n_samples]
    labels = labels[:n_samples]

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig.suptitle("Sample Training Images (after augmentation)", fontsize=16, y=1.01)

    for idx, ax in enumerate(axes.flatten()):
        if idx >= len(images):
            ax.axis("off")
            continue

        # Denormalize for display
        img = images[idx].numpy().transpose(1, 2, 0)  # (C,H,W) → (H,W,C)
        img = img * IMAGENET_STD + IMAGENET_MEAN
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        # Shorten label for display
        label = class_names[labels[idx].item()].replace("___", "\n")
        ax.set_title(label, fontsize=8, pad=4)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n  [✓] Saved sample grid to 'sample_images.png'")


# ==============================================================
# STEP 7 — Plot class distribution (check for imbalance)
# ==============================================================

def plot_class_distribution(class_data: dict):
    """
    Bar chart showing number of images per class.
    Helps identify imbalanced classes that may need special handling.
    """
    class_names = list(class_data.keys())
    counts = [len(v) for v in class_data.values()]

    # Shorten names for readability
    short_names = [n.split("___")[-1] for n in class_names]

    fig, ax = plt.subplots(figsize=(18, 6))
    bars = ax.bar(range(len(counts)), counts, color="#4CAF50", edgecolor="white", width=0.7)
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution — PlantVillage Dataset", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  [✓] Saved distribution chart to 'class_distribution.png'")


# ==============================================================
# MAIN — Run everything in order
# ==============================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  PLANT DISEASE DETECTION — PHASE 1: Data Preparation")
    print("="*60)

    # 1. Explore raw dataset
    class_data = explore_dataset(DATASET_PATH)

    # 2. Plot class distribution
    plot_class_distribution(class_data)

    # 3. Split into train/val/test
    split_dataset(class_data, SPLIT_OUTPUT_PATH, seed=RANDOM_SEED)

    # 4. Create DataLoaders
    dataloaders, class_names = get_dataloaders(SPLIT_OUTPUT_PATH)

    # 5. Save class names for inference later
    save_class_names(class_names)

    # 6. Visualize sample images
    visualize_samples(dataloaders, class_names)

    print("\n" + "="*60)
    print("  PHASE 1 COMPLETE! Next step: Phase 2 → Model Architecture")
    print("="*60 + "\n")