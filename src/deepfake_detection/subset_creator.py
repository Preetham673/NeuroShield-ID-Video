"""
subset_creator.py

Create a smaller balanced dataset
for fast development and debugging.
"""

import random
import shutil
from pathlib import Path

random.seed(42)

SOURCE = Path(r"D:\NeuroShield_Research\processed_data")

DESTINATION = Path(r"D:\NeuroShield_Research\processed_data_subset")

IMAGES_PER_CLASS = 5000


def copy_subset(split):

    real_source = SOURCE / split / "real"
    fake_source = SOURCE / split / "fake"

    real_dest = DESTINATION / split / "real"
    fake_dest = DESTINATION / split / "fake"

    real_dest.mkdir(parents=True, exist_ok=True)
    fake_dest.mkdir(parents=True, exist_ok=True)

    real_images = list(real_source.glob("*.jpg"))
    fake_images = list(fake_source.glob("*.jpg"))

    random.shuffle(real_images)
    random.shuffle(fake_images)

    real_images = real_images[:min(IMAGES_PER_CLASS, len(real_images))]
    fake_images = fake_images[:min(IMAGES_PER_CLASS, len(fake_images))]

    print(f"\nCopying {split} dataset...")

    print(f"Real : {len(real_images)}")
    print(f"Fake : {len(fake_images)}")

    for image in real_images:
        shutil.copy(image, real_dest / image.name)

    for image in fake_images:
        shutil.copy(image, fake_dest / image.name)


def main():

    copy_subset("train")
    copy_subset("validation")
    copy_subset("test")

    print("\nSubset Created Successfully!")


if __name__ == "__main__":
    main()