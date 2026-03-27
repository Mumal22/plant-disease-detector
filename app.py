"""
==============================================================
  PLANT DISEASE DETECTION — PHASE 6: Flask Web App (Backend)
==============================================================
  What this file does:
    1. Loads the trained model (best_model.pth)
    2. Serves the HTML frontend (templates/index.html)
    3. Exposes a /predict API endpoint:
         - Accepts an uploaded leaf image
         - Preprocesses it
         - Runs inference
         - Returns top-3 predictions + confidence scores as JSON
    4. Exposes a /health endpoint for status checks

  Run with:
    python app.py

  Then open: http://localhost:5000
==============================================================
"""

import io
import os
import json
import base64

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from flask import Flask, request, jsonify, render_template

from model import build_model


# ==============================================================
# CONFIGURATION
# ==============================================================

MODEL_PATH       = "best_model.pth"
CLASS_NAMES_PATH = "class_names.json"
NUM_CLASSES      = 38
IMAGE_SIZE       = 224
MAX_FILE_SIZE_MB = 10

# Treatment suggestions per disease keyword
# Add more as needed
TREATMENT_TIPS = {
    "healthy":        "No treatment needed. Plant looks healthy!",
    "early_blight":   "Apply copper-based fungicide. Remove affected leaves.",
    "late_blight":    "Use chlorothalonil fungicide. Improve air circulation.",
    "leaf_mold":      "Reduce humidity. Apply fungicide spray.",
    "mosaic_virus":   "Remove infected plants. Control aphid vectors.",
    "rust":           "Apply sulfur-based fungicide. Avoid overhead watering.",
    "scab":           "Apply fungicide at bud break. Prune for air flow.",
    "black_rot":      "Remove infected tissue. Apply copper fungicide.",
    "cercospora":     "Apply fungicide. Avoid working with wet plants.",
    "powdery_mildew": "Apply potassium bicarbonate or neem oil spray.",
    "spider_mites":   "Use miticide or neem oil. Increase humidity.",
    "target_spot":    "Apply fungicide. Remove heavily infected leaves.",
    "yellow_leaf_curl": "Control whitefly population. Remove infected plants.",
    "bacterial_spot": "Apply copper-based bactericide. Avoid overhead irrigation.",
    "greening":       "Remove infected trees. Control psyllid insects.",
    "haunglongbing":  "No cure available. Remove infected trees immediately.",
}


def get_treatment(class_name: str) -> str:
    """Returns a treatment suggestion based on disease keywords in class name."""
    class_lower = class_name.lower()
    for keyword, tip in TREATMENT_TIPS.items():
        if keyword in class_lower:
            return tip
    return "Consult a local agricultural extension officer for treatment advice."


# ==============================================================
# APP SETUP
# ==============================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024

# Load model and class names once at startup
print("\n  Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

with open(CLASS_NAMES_PATH) as f:
    CLASS_NAMES = json.load(f)

model = build_model(
    num_classes=NUM_CLASSES,
    weights_path=MODEL_PATH,
    device="cpu"
)
model.eval()
torch.set_grad_enabled(False)


# Image preprocessing pipeline (same as validation transform in Phase 1)
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ==============================================================
# ROUTES
# ==============================================================

@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "device": device,
        "num_classes": NUM_CLASSES,
        "model": MODEL_PATH
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts: multipart/form-data with field 'file' (image)
    Returns: JSON with top-3 predictions and treatment tip

    Response format:
    {
        "success": true,
        "predictions": [
            {"rank": 1, "class": "Tomato___Early_blight", "confidence": 94.2, "treatment": "..."},
            {"rank": 2, ...},
            {"rank": 3, ...}
        ],
        "plant":   "Tomato",
        "disease": "Early blight",
        "is_healthy": false
    }
    """
    # --- Validate request ---
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    allowed = {"jpg", "jpeg", "png", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return jsonify({"success": False,
                        "error": f"Invalid file type. Allowed: {allowed}"}), 400

    try:
        # --- Load and preprocess image ---
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor    = TRANSFORM(pil_img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

        # --- Run inference ---
        with torch.no_grad():
            output = model(tensor)
            probs  = F.softmax(output, dim=1)[0]   # (38,)

        # --- Top-3 results ---
        top3_probs, top3_idx = torch.topk(probs, k=3)

        predictions = []
        for rank, (idx, prob) in enumerate(zip(top3_idx, top3_probs), start=1):
            class_name = CLASS_NAMES[idx.item()]
            confidence = round(prob.item() * 100, 2)
            predictions.append({
                "rank":       rank,
                "class":      class_name,
                "confidence": confidence,
                "treatment":  get_treatment(class_name)
            })

        # Parse plant and disease from top prediction
        top_class  = predictions[0]["class"]
        parts      = top_class.split("___")
        plant      = parts[0].replace("_", " ") if len(parts) > 0 else "Unknown"
        disease    = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        is_healthy = "healthy" in top_class.lower()

        return jsonify({
            "success":    True,
            "predictions": predictions,
            "plant":      plant,
            "disease":    disease,
            "is_healthy": is_healthy
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Plant Disease Detection — Web App")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)