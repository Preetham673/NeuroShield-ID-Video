"""
inference.py

Predict whether an image is
Real or Fake.
"""

import torch
from PIL import Image

from src.deepfake_detection.model import build_model, DEVICE
from src.deepfake_detection.transforms import test_transform

# ------------------------------------
# Load Trained Model
# ------------------------------------

model = build_model()

model.load_state_dict(
    torch.load(
        "models/deepfake_detector.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

# ------------------------------------

classes = ["REAL", "FAKE"]


def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = test_transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(output, dim=1)

        confidence, prediction = torch.max(probabilities, 1)

    prediction_label = classes[prediction.item()]

    confidence_score = confidence.item() * 100

    print("\n========== DEEPFAKE DETECTION ==========\n")

    print(f"Prediction : {prediction_label}")

    print(f"Confidence : {confidence_score:.2f}%")

    return {
        "prediction": prediction_label,
        "confidence": confidence_score
    }