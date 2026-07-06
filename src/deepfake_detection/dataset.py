"""
dataset.py

Custom PyTorch Dataset for NeuroShield-ID
Deepfake Detection Module.
"""

from pathlib import Path

from PIL import Image

from torch.utils.data import Dataset

from src.deepfake_detection.transforms import (
    train_transform,
    test_transform,
)

# ==========================================
# Dataset Class
# ==========================================

class DeepfakeDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = Path(root_dir)

        self.transform = transform

        self.image_paths = []

        self.labels = []

        # --------------------------
        # Real Images
        # --------------------------

        real_dir = self.root_dir / "real"

        for image in real_dir.glob("*.jpg"):

            self.image_paths.append(image)

            self.labels.append(0)

        # --------------------------
        # Fake Images
        # --------------------------

        fake_dir = self.root_dir / "fake"

        for image in fake_dir.glob("*.jpg"):

            self.image_paths.append(image)

            self.labels.append(1)

        print(f"\nLoaded {len(self.image_paths)} images.")

    # ======================================

    def __len__(self):

        return len(self.image_paths)

    # ======================================

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index]).convert("RGB")

        label = self.labels[index]

        if self.transform:

            image = self.transform(image)

        return image, label