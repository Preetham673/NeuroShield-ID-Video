"""
model.py

EfficientNet-B0 model for
Deepfake Detection.
"""

import torch
import torch.nn as nn
import timm

# ==========================================
# Create Model
# ==========================================

def build_model():

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True
    )

    in_features = model.classifier.in_features

    model.classifier = nn.Linear(
        in_features,
        2
    )

    return model


# ==========================================
# Device
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)