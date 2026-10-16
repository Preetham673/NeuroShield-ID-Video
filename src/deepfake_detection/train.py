"""
train.py

Main training script
"""

import torch

from src.deepfake_detection.trainer import (
    train_one_epoch,
    validate,
    model,
    EPOCHS,
)

from src.deepfake_detection.model import MODEL_PATH


def main():

    best_accuracy = -1.0

    for epoch in range(1, EPOCHS + 1):

        loss = train_one_epoch(epoch)

        accuracy = validate()

        print()

        print(f"Epoch {epoch}")

        print(f"Loss: {loss:.4f}")

        print(f"Validation Accuracy: {accuracy:.2f}%")

        # Keep the best epoch by validation accuracy. The old script saved
        # only the final epoch, which may be worse than an earlier one.
        if accuracy > best_accuracy:

            best_accuracy = accuracy

            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), MODEL_PATH)

            print("Saved new best model.")

    print()

    print(f"Training finished. Best validation accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()