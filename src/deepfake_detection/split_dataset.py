"""
split_dataset.py

Split extracted images into
train, validation and test sets.
"""

import random
import shutil
from pathlib import Path

# ==========================================
# Dataset Paths
# ==========================================

DATASET_ROOT = Path(r"D:\NeuroShield_Research\processed_data")

REAL_DIR = DATASET_ROOT / "train" / "real"
FAKE_DIR = DATASET_ROOT / "train" / "fake"

VAL_REAL = DATASET_ROOT / "validation" / "real"
VAL_FAKE = DATASET_ROOT / "validation" / "fake"

TEST_REAL = DATASET_ROOT / "test" / "real"
TEST_FAKE = DATASET_ROOT / "test" / "fake"

# ==========================================
# Split Ratios
# ==========================================

VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


def split_images(source_dir, val_dir, test_dir):

    images = list(source_dir.glob("*.jpg"))

    random.shuffle(images)

    total_images = len(images)

    val_count = int(total_images * VALIDATION_RATIO)
    test_count = int(total_images * TEST_RATIO)

    validation_images = images[:val_count]
    test_images = images[val_count:val_count + test_count]

    print(f"\nProcessing {source_dir.name}")
    print(f"Total Images : {total_images}")
    print(f"Validation   : {len(validation_images)}")
    print(f"Test         : {len(test_images)}")

    for image in validation_images:
        shutil.copy(image, val_dir / image.name)

    for image in test_images:
        shutil.copy(image, test_dir / image.name)


def main():

    split_images(REAL_DIR, VAL_REAL, TEST_REAL)

    split_images(FAKE_DIR, VAL_FAKE, TEST_FAKE)

    print("\nDataset Split Completed Successfully!")


if __name__ == "__main__":
    main()