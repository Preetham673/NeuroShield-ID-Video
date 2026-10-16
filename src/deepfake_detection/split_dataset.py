"""
split_dataset.py

Split extracted images into train, validation and test sets.

Splits BY VIDEO, not by individual frame, and MOVES files out of train.

Why: frames are extracted every 10th frame, so neighbouring frames of one
video are nearly identical. Splitting by frame puts near-duplicates of the
same clip into train and test, and copying (instead of moving) also left
every validation/test image inside train. Both make the reported accuracy
look far better than it really is.
"""

import os
import random
import shutil
import sys
from pathlib import Path

# ==========================================
# Dataset Paths
# ==========================================

DATASET_ROOT = Path(
    os.environ.get("NEUROSHIELD_RAW_ROOT", r"D:\NeuroShield_Research\processed_data")
)

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

SEED = 42


def video_id(image_path):
    # frame_extractor names frames "<video_stem>_<frame_no>.jpg"
    return image_path.stem.rsplit("_", 1)[0]


def split_images(source_dir, val_dir, test_dir):

    images = list(source_dir.glob("*.jpg"))

    groups = {}
    for image in images:
        groups.setdefault(video_id(image), []).append(image)

    videos = sorted(groups)
    random.Random(SEED).shuffle(videos)

    total_images = len(images)
    val_target = int(total_images * VALIDATION_RATIO)
    test_target = int(total_images * TEST_RATIO)

    validation_images, test_images = [], []

    for vid in videos:
        if len(validation_images) < val_target:
            validation_images.extend(groups[vid])
        elif len(test_images) < test_target:
            test_images.extend(groups[vid])
        else:
            break

    print(f"\nProcessing {source_dir.name}")
    print(f"Videos       : {len(videos)}")
    print(f"Total Images : {total_images}")
    print(f"Validation   : {len(validation_images)}")
    print(f"Test         : {len(test_images)}")

    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for image in validation_images:
        shutil.move(str(image), str(val_dir / image.name))

    for image in test_images:
        shutil.move(str(image), str(test_dir / image.name))


def main():

    # Refuse to run on top of an old (leaky) split.
    for d in (VAL_REAL, VAL_FAKE, TEST_REAL, TEST_FAKE):
        if d.exists() and any(d.iterdir()):
            print(f"{d} is not empty. Delete the old validation/ and test/ "
                  "folders first, then re-run (train/ still holds every image).")
            sys.exit(1)

    split_images(REAL_DIR, VAL_REAL, TEST_REAL)

    split_images(FAKE_DIR, VAL_FAKE, TEST_FAKE)

    print("\nDataset Split Completed Successfully!")


if __name__ == "__main__":
    main()