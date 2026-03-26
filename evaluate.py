"""
==============================================================
  PLANT DISEASE DETECTION — PHASE 5: Evaluation & Testing
==============================================================
  What this file does:
    1. Loads the best saved model (best_model.pth)
    2. Evaluates on the held-out test set
    3. Prints accuracy, precision, recall, F1-score per class
    4. Generates a confusion matrix heatmap
    5. Grad-CAM: visualizes which leaf regions triggered the prediction
    6. Shows sample predictions with confidence scores
==============================================================
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from model import build_model
from data_prep import get_dataloaders, SPLIT_OUTPUT_PATH


# ==============================================================
# CONFIGURATION
# ==============================================================

MODEL_PATH      = "best_model.pth"
CLASS_NAMES_PATH = "class_names.json"
NUM_CLASSES     = 38
IMAGE_SIZE      = 224


# ==============================================================
# STEP 1 — Load model and class names
# ==============================================================

def load_model_and_classes():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)

    model = build_model(
        num_classes=NUM_CLASSES,
        weights_path=MODEL_PATH,
        device=device
    )
    model.eval()

    print(f"  Device      : {device.upper()}")
    print(f"  Model loaded: {MODEL_PATH}")
    print(f"  Classes     : {len(class_names)}")
    return model, class_names, device


# ==============================================================
# STEP 2 — Evaluate on test set
# ==============================================================

def evaluate_test_set(model, dataloader, class_names, device):
    """
    Runs model on every test image and collects:
      - all true labels
      - all predicted labels
      - all confidence scores
    Then prints full classification report.
    """
    model.eval()

    all_preds  = []
    all_labels = []
    all_probs  = []

    print("\n  Running inference on test set...")

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs   = F.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    acc = accuracy_score(all_labels, all_preds) * 100
    print(f"\n  Test Accuracy: {acc:.2f}%")

    # Per-class report
    print("\n" + "="*60)
    print("  PER-CLASS REPORT")
    print("="*60)
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=3
    ))

    return all_preds, all_labels, np.array(all_probs)


# ==============================================================
# STEP 3 — Confusion matrix
# ==============================================================

def plot_confusion_matrix(all_labels, all_preds, class_names):
    """
    Plots a heatmap of the confusion matrix.
    Diagonal = correct predictions (want these bright).
    Off-diagonal = mistakes (want these dark).
    """
    cm = confusion_matrix(all_labels, all_preds)

    # Normalize to percentages
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # Short names for readability
    short_names = [n.split("___")[-1][:15] for n in class_names]

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm_norm,
        annot=True, fmt=".0f",
        xticklabels=short_names,
        yticklabels=short_names,
        cmap="YlGn",
        linewidths=0.3,
        ax=ax,
        cbar_kws={"label": "% predicted as class"}
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Confusion Matrix (%) — PlantDiseaseNet", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  [✓] Saved confusion matrix to 'confusion_matrix.png'")


# ==============================================================
# STEP 4 — Grad-CAM implementation
# ==============================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)

    What it does:
      - Shows WHICH parts of the leaf image the CNN focused on
        when making its disease prediction
      - Produces a heatmap overlay on the original image
      - Helps verify the model is looking at the right regions
        (diseased spots) rather than background noise

    How it works:
      1. Register hooks on the last conv layer to capture:
         - The feature maps (forward hook)
         - The gradients flowing back (backward hook)
      2. Run forward pass → get prediction
      3. Run backward pass on the predicted class score
      4. Weight each feature map channel by its average gradient
      5. ReLU the result → only positive contributions matter
      6. Resize heatmap to original image size
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Generates Grad-CAM heatmap for input_tensor.
        If class_idx is None, uses the top predicted class.
        Returns: numpy heatmap of shape (H, W) in range [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass on the target class score
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Pool gradients over spatial dimensions → (channels,)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Only keep positive influence

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, class_idx


def visualize_gradcam(model, class_names, device, num_images=6):
    """
    Runs Grad-CAM on sample test images and shows:
      - Original image
      - Grad-CAM heatmap overlay
      - Predicted class + confidence
    """
    # Target the last conv layer in the last ConvBlock
    # This is where the most abstract, disease-specific features live
    target_layer = model.features[-1].block[0]  # Last Conv2d
    grad_cam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Collect sample images from test set
    test_path = Path(SPLIT_OUTPUT_PATH) / "test"
    sample_images = []
    sample_labels = []

    with open(CLASS_NAMES_PATH) as f:
        class_names_list = json.load(f)

    # Pick one image from each of 6 random classes
    import random
    random.seed(42)
    class_folders = sorted(test_path.iterdir())
    chosen = random.sample(class_folders, min(num_images, len(class_folders)))

    for folder in chosen:
        imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.JPG")) + list(folder.glob("*.png"))
        if imgs:
            sample_images.append(imgs[0])
            sample_labels.append(folder.name)

    fig, axes = plt.subplots(len(sample_images), 3, figsize=(12, 4 * len(sample_images)))
    fig.suptitle("Grad-CAM Visualization — What the model sees", fontsize=14, y=1.01)

    for idx, (img_path, true_label) in enumerate(zip(sample_images, sample_labels)):

        # Load and preprocess
        pil_img   = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Generate Grad-CAM
        heatmap, pred_idx = grad_cam.generate(input_tensor)
        pred_label = class_names_list[pred_idx]

        # Get confidence
        with torch.no_grad():
            out   = model(input_tensor)
            probs = F.softmax(out, dim=1)
            conf  = probs[0, pred_idx].item() * 100

        # Resize heatmap to image size
        heatmap_resized = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
            )
        ) / 255.0

        # Create overlay
        colormap  = cm.get_cmap("jet")
        heatmap_color = colormap(heatmap_resized)[:, :, :3]
        orig_array    = np.array(pil_img) / 255.0
        overlay       = 0.55 * orig_array + 0.45 * heatmap_color
        overlay       = np.clip(overlay, 0, 1)

        row = axes[idx] if len(sample_images) > 1 else axes

        # Col 0: Original
        row[0].imshow(orig_array)
        row[0].set_title(f"True: {true_label.split('___')[-1]}", fontsize=9)
        row[0].axis("off")

        # Col 1: Heatmap only
        row[1].imshow(heatmap_resized, cmap="jet")
        row[1].set_title("Grad-CAM heatmap", fontsize=9)
        row[1].axis("off")

        # Col 2: Overlay
        correct = "✓" if pred_label == true_label else "✗"
        row[2].imshow(overlay)
        row[2].set_title(
            f"{correct} Pred: {pred_label.split('___')[-1]}\n({conf:.1f}%)",
            fontsize=9,
            color="green" if pred_label == true_label else "red"
        )
        row[2].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  [✓] Saved Grad-CAM results to 'gradcam_results.png'")


# ==============================================================
# STEP 5 — Predict a single image (useful for Phase 6 / API)
# ==============================================================

def predict_single_image(image_path: str, model, class_names: list, device: str):
    """
    Predicts the disease class for a single image file.
    Returns the top-3 predictions with confidence scores.
    This function will be reused in the Flask/FastAPI app (Phase 6).
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)  # Add batch dim

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs  = F.softmax(output, dim=1)[0]

    # Top-3 predictions
    top3_probs, top3_idx = torch.topk(probs, k=3)

    print(f"\n  Image: {image_path}")
    print("  Top-3 Predictions:")
    for i, (idx, prob) in enumerate(zip(top3_idx, top3_probs)):
        print(f"    {i+1}. {class_names[idx.item()]:<45}  {prob.item()*100:.1f}%")

    return {
        "predicted_class": class_names[top3_idx[0].item()],
        "confidence":      top3_probs[0].item() * 100,
        "top3": [
            {"class": class_names[idx.item()], "confidence": prob.item() * 100}
            for idx, prob in zip(top3_idx, top3_probs)
        ]
    }


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  PLANT DISEASE DETECTION — PHASE 5: Evaluation")
    print("="*60)

    # 1. Load model
    model, class_names, device = load_model_and_classes()

    # 2. Load test dataloader
    dataloaders, _ = get_dataloaders(SPLIT_OUTPUT_PATH)

    # 3. Evaluate on test set
    all_preds, all_labels, all_probs = evaluate_test_set(
        model, dataloaders["test"], class_names, device
    )

    # 4. Confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names)

    # 5. Grad-CAM visualization
    print("\n  Generating Grad-CAM visualizations...")
    visualize_gradcam(model, class_names, device, num_images=6)

    print("\n" + "="*60)
    print("  PHASE 5 COMPLETE!")
    print("  Files saved: confusion_matrix.png, gradcam_results.png")
    print("  Next step  : Phase 6 → Deploy as web app")
    print("="*60 + "\n")