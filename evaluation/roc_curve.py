"""
roc_curve.py

Generate ROC Curve for the trained model.
"""

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
)

from src.deepfake_detection.model import (
    build_model,
    DEVICE,
)

from src.deepfake_detection.dataloader import (
    test_loader,
)

# ---------------------------------
# Load Model
# ---------------------------------

print("Loading Model...")

model = build_model()

model.load_state_dict(
    torch.load(
        "models/deepfake_detector.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()

# ---------------------------------
# Prediction
# ---------------------------------

true_labels = []
probabilities = []

print("Running Inference...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        probabilities.extend(
            probs[:, 1].cpu().numpy()
        )

        true_labels.extend(
            labels.numpy()
        )

# ---------------------------------
# ROC
# ---------------------------------

fpr, tpr, thresholds = roc_curve(
    true_labels,
    probabilities
)

roc_auc = auc(
    fpr,
    tpr
)

# ---------------------------------
# Plot
# ---------------------------------

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}",
    linewidth=2
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.savefig(
    "roc_curve.png",
    dpi=300
)

plt.show()

print()

print(f"ROC AUC : {roc_auc:.4f}")

print("ROC Curve saved as roc_curve.png")