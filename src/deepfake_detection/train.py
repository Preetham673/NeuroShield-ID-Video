"""
train.py

Main training script
"""

import torch
import os

from src.deepfake_detection.trainer import (
    train_one_epoch,
    validate,
    model,
    EPOCHS,
)

for epoch in range(1, EPOCHS + 1):

    loss = train_one_epoch(epoch)

    accuracy = validate()

    print()

    print(f"Epoch {epoch}")

    print(f"Loss: {loss:.4f}")

    print(f"Validation Accuracy: {accuracy:.2f}%")

os.makedirs("models", exist_ok=True)
torch.save(
    model.state_dict(),
    "models/deepfake_detector.pth"
)

print()

print("Model Saved Successfully!")