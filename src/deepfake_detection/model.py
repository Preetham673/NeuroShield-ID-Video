"""
model.py

EfficientNet-B0 model for
Deepfake Detection.
"""

from pathlib import Path

import torch
import torch.nn as nn
import timm

# <root>/src/deepfake_detection/model.py -> <root>
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Single source of truth for the trained weights, so it no longer
# depends on which folder the app/script is launched from.
MODEL_PATH = PROJECT_DIR / "models" / "deepfake_detector.pth"

# ==========================================
# Create Model
# ==========================================

def build_model(pretrained=True):
    """
    pretrained=True  -> for training (ImageNet weights as a starting point)
    pretrained=False -> when loading our own trained weights (inference,
                        evaluation, Grad-CAM). Skips a needless ImageNet
                        download that load_state_dict overwrites anyway
                        and that fails when offline.
    """

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained
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