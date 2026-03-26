"""
==============================================================
  PLANT DISEASE DETECTION — PHASE 2: CNN Model Architecture
==============================================================
  What this file does:
    1. Defines a custom CNN built from scratch in PyTorch
    2. Architecture: 3 Conv Blocks → Flatten → FC Layers
    3. Each Conv Block: Conv2d → BatchNorm → ReLU → MaxPool
    4. Classifier head: FC(512) → Dropout → FC(num_classes)
    5. Includes a model summary utility function
    6. Includes a quick test to verify forward pass works
==============================================================
"""

import torch
import torch.nn as nn


# ==============================================================
# THE CNN MODEL — PlantDiseaseNet
# ==============================================================

class ConvBlock(nn.Module):
    """
    A single Convolutional Block:
      Conv2d → BatchNorm2d → ReLU → MaxPool2d

    WHY each layer:
      - Conv2d       : Learns spatial features (edges, textures, patterns)
      - BatchNorm2d  : Normalizes activations → faster, more stable training
      - ReLU         : Non-linearity so the model can learn complex patterns
      - MaxPool2d    : Reduces spatial size by 2x, keeps strongest features
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            # Conv2d(in, out, kernel_size, padding)
            # padding=1 keeps output size same as input before pooling
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # halves H and W
        )

    def forward(self, x):
        return self.block(x)


class PlantDiseaseNet(nn.Module):
    """
    Custom CNN for Plant Disease Classification

    Architecture overview:
      Input:  (batch, 3, 224, 224)   — RGB image

      Block 1: 3  → 32  filters,  output: (batch, 32,  112, 112)
      Block 2: 32 → 64  filters,  output: (batch, 64,   56,  56)
      Block 3: 64 → 128 filters,  output: (batch, 128,  28,  28)

      Adaptive pool:               output: (batch, 128,   8,   8)

      Flatten:                     output: (batch, 128 * 8 * 8 = 8192)

      FC1:  8192 → 512 + ReLU
      Dropout(0.5)
      FC2:  512  → num_classes     (38 for PlantVillage)

    Design decisions:
      - Filters double each block (32→64→128) to learn richer features
      - AdaptiveAvgPool makes the model flexible to different image sizes
      - Dropout 0.5 prevents overfitting
    """

    def __init__(self, num_classes: int = 38):
        super(PlantDiseaseNet, self).__init__()

        # ---- Feature Extractor (3 Conv Blocks) ----
        self.features = nn.Sequential(
            ConvBlock(in_channels=3,   out_channels=32),   # RGB → 32 filters
            ConvBlock(in_channels=32,  out_channels=64),   # 32 → 64 filters
            ConvBlock(in_channels=64,  out_channels=128),  # 64 → 128 filters
        )

        # ---- Adaptive pooling (makes output always 8x8) ----
        # This means if someone passes a 256x256 image it still works
        self.pool = nn.AdaptiveAvgPool2d(output_size=(8, 8))

        # ---- Classifier Head ----
        self.classifier = nn.Sequential(
            nn.Flatten(),                         # (batch, 128, 8, 8) → (batch, 8192)
            nn.Linear(128 * 8 * 8, 512),         # FC layer: 8192 → 512
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),                    # Randomly zeros 50% of neurons
            nn.Linear(512, num_classes),          # FC layer: 512 → 38 classes
            # NOTE: No Softmax here — CrossEntropyLoss includes it internally
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          x: input tensor of shape (batch_size, 3, 224, 224)
          returns: logits of shape (batch_size, num_classes)
        """
        x = self.features(x)   # Extract features
        x = self.pool(x)        # Reduce to 8x8
        x = self.classifier(x) # Classify
        return x


# ==============================================================
# UTILITY — Print model summary
# ==============================================================

def get_model_summary(model: nn.Module, input_size=(1, 3, 224, 224)):
    """
    Prints a summary of the model:
      - Layer names and types
      - Output shape at each layer
      - Total and trainable parameter count
    """
    print("\n" + "="*60)
    print("  MODEL SUMMARY — PlantDiseaseNet")
    print("="*60)

    device = next(model.parameters()).device
    x = torch.zeros(input_size).to(device)

    print(f"\n  Input shape  : {tuple(x.shape)}")

    # Pass through feature extractor block by block
    for i, block in enumerate(model.features):
        x = block(x)
        print(f"  Conv block {i+1} : {tuple(x.shape)}")

    x = model.pool(x)
    print(f"  After pool   : {tuple(x.shape)}")

    x = model.classifier(x)
    print(f"  Output shape : {tuple(x.shape)}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable:,}")
    print("="*60 + "\n")


# ==============================================================
# UTILITY — Build and optionally load weights
# ==============================================================

def build_model(num_classes: int = 38,
                weights_path: str = None,
                device: str = None) -> PlantDiseaseNet:
    """
    Creates and returns a PlantDiseaseNet model.

    Args:
        num_classes  : Number of disease classes (38 for PlantVillage)
        weights_path : Path to saved .pth weights file (optional)
        device       : 'cuda', 'cpu', or None (auto-detect)

    Returns:
        model on the appropriate device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PlantDiseaseNet(num_classes=num_classes)
    model = model.to(device)

    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"  [✓] Loaded weights from: {weights_path}")

    return model


# ==============================================================
# QUICK TEST — Verify the model works with a dummy input
# ==============================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  PLANT DISEASE DETECTION — PHASE 2: Model Architecture")
    print("="*60)

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device.upper()}")

    # Build model
    NUM_CLASSES = 38  # PlantVillage has 38 disease/healthy classes
    model = build_model(num_classes=NUM_CLASSES, device=device)

    # Print summary
    get_model_summary(model)

    # Forward pass test with a fake batch of 4 images
    print("  Running forward pass test...")
    dummy_input = torch.randn(4, 3, 224, 224).to(device)  # batch of 4 images
    output = model(dummy_input)

    print(f"  Input  : {tuple(dummy_input.shape)}")
    print(f"  Output : {tuple(output.shape)}  (batch=4, classes=38)")

    # Verify output shape is correct
    assert output.shape == (4, NUM_CLASSES), "ERROR: Output shape mismatch!"
    print("\n  [✓] Forward pass test PASSED!")

    # Show parameter count
    total = sum(p.numel() for p in model.parameters())
    print(f"  [✓] Model has {total:,} parameters")

    print("\n" + "="*60)
    print("  PHASE 2 COMPLETE! Next step: Phase 3 → Training")
    print("="*60 + "\n")