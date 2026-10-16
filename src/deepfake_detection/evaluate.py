"""
evaluate.py

Evaluate trained EfficientNet-B0 model.
"""

import time
import torch
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.deepfake_detection.model import (
    build_model,
    DEVICE,
    MODEL_PATH,
)

from src.deepfake_detection.dataloader import (
    test_loader,
)


def evaluate_model(show_output=True):
    """
    Evaluate the trained model and return all metrics.
    """

    if show_output:
        print("\nLoading trained model...\n")

    # ---------------------------------
    # Load Model
    # ---------------------------------

    model = build_model(pretrained=False)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    model.to(DEVICE)
    model.eval()

    # ---------------------------------
    # Prediction
    # ---------------------------------

    true_labels = []
    predicted_labels = []
    probabilities = []

    start_time = time.time()

    # One pass over the test set collects predictions AND probabilities.
    # The old code looped over the whole test set twice (again for the
    # ROC probabilities), doubling the wait on the Streamlit pages.
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            predicted = probs.argmax(dim=1)

            true_labels.extend(labels.numpy())

            predicted_labels.extend(predicted.cpu().numpy())

            probabilities.extend(probs[:, 1].cpu().numpy())

    end_time = time.time()

    # ---------------------------------
    # Metrics
    # ---------------------------------

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels
    )

    recall = recall_score(
        true_labels,
        predicted_labels
    )

    f1 = f1_score(
        true_labels,
        predicted_labels
    )

    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )


    # ROC Curve

    fpr, tpr, _ = roc_curve(
    true_labels,
    probabilities
)

    roc_auc = auc(
    fpr,
    tpr
)

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=["REAL", "FAKE"],
        output_dict=True
    )

    results = {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "confusion_matrix": cm,

        "classification_report": report,

        "true_labels": true_labels,

        "predicted_labels": predicted_labels,

        "inference_time": end_time - start_time,

        "fpr": fpr,

        "tpr": tpr,

        "roc_auc": roc_auc,

        "num_samples": len(true_labels),

    }

    if show_output:

        print("\n===================================")
        print("      MODEL EVALUATION")
        print("===================================\n")

        print(f"Accuracy      : {accuracy*100:.2f}%")
        print(f"Precision     : {precision*100:.2f}%")
        print(f"Recall        : {recall*100:.2f}%")
        print(f"F1 Score      : {f1*100:.2f}%")
        print(f"Inference Time: {end_time-start_time:.2f} sec")

        print("\n========== CONFUSION MATRIX ==========\n")

        print(cm)

        print("\n========== CLASSIFICATION REPORT ==========\n")

        print(
            classification_report(
                true_labels,
                predicted_labels,
                target_names=["REAL", "FAKE"]
            )
        )

    return results


if __name__ == "__main__":

    evaluate_model(show_output=True)