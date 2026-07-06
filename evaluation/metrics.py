"""
metrics.py

Evaluate the trained Deepfake Detection model.
"""

import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from src.deepfake_detection.model import (
    build_model,
    DEVICE,
)

from src.deepfake_detection.validation_loader import validation_loader

# -----------------------------
# Load Model
# -----------------------------

model = build_model()

model.load_state_dict(
    torch.load(
        "models/deepfake_detector.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()


def evaluate():

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.numpy())

            y_pred.extend(
                predictions.cpu().numpy()
            )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    print("\n========== MODEL EVALUATION ==========\n")

    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")

    print("\nClassification Report\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["REAL", "FAKE"]
        )
    )


if __name__ == "__main__":

    evaluate()