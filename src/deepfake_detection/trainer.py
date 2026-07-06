"""
trainer.py

Training engine for EfficientNet-B0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.deepfake_detection.model import build_model, DEVICE
from src.deepfake_detection.dataloader import (
    train_loader,
    validation_loader,
)

# ==========================================
# Hyperparameters
# ==========================================

LEARNING_RATE = 1e-4
EPOCHS = 3

# ==========================================
# Model
# ==========================================

model = build_model()
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4

)

# ==========================================
# Training Function
# ==========================================

def train_one_epoch(epoch):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    progress_bar = tqdm(train_loader)

    for images, labels in progress_bar:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

        accuracy = 100 * correct / total

        progress_bar.set_description(
            f"Epoch {epoch}"
        )

        progress_bar.set_postfix(
            loss=loss.item(),
            accuracy=f"{accuracy:.2f}%"
        )

    return running_loss / len(train_loader)

# ==========================================
# Validation
# ==========================================

def validate():

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy