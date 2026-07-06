"""
dataloader.py

PyTorch DataLoader
for Deepfake Detection.
"""

from torch.utils.data import DataLoader

from src.deepfake_detection.dataset import DeepfakeDataset
from src.deepfake_detection.transforms import (
    train_transform,
    test_transform,
)

TRAIN_PATH = r"D:\NeuroShield_Research\processed_data_subset\train"

VALIDATION_PATH = r"D:\NeuroShield_Research\processed_data_subset\validation"

TEST_PATH = r"D:\NeuroShield_Research\processed_data_subset\test"

# ==========================================
# Train Loader
# ==========================================

train_dataset = DeepfakeDataset(
    TRAIN_PATH,
    transform=train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

# ==========================================
# Validation Loader
# ==========================================

validation_dataset = DeepfakeDataset(
    VALIDATION_PATH,
    transform=test_transform
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# ==========================================
# Test Loader
# ==========================================

test_dataset = DeepfakeDataset(
    TEST_PATH,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)