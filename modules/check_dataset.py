from PIL import Image
import os
import sys
import argparse

# Was hardcoded to a Windows path on one machine (D:\NeuroShield_Research\...),
# which meant the script only ran there. Now it takes the path as an
# argument, falling back to an env var, and finally that old default so
# nothing breaks if you still run it the old way.
parser = argparse.ArgumentParser(description="Scan a dataset folder for corrupted images.")
parser.add_argument(
    "dataset_path",
    nargs="?",
    default=os.environ.get("NEUROSHIELD_DATASET_PATH", r"D:\NeuroShield_Research\processed_data_subset\test"),
    help="Path to the dataset folder to scan (defaults to $NEUROSHIELD_DATASET_PATH).",
)
args = parser.parse_args()
dataset_path = args.dataset_path

if not os.path.isdir(dataset_path):
    print(f"Dataset path does not exist or is not a directory: {dataset_path}")
    sys.exit(1)

bad_files = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        path = os.path.join(root, file)
        try:
            # verify() only checks file/header consistency; it does not
            # decode pixel data, so a truncated JPEG can pass verify()
            # and only fail later, mid-training. Reopening and calling
            # load() forces a full decode so truncation is actually caught.
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.load()
        except Exception as e:
            print(f"Corrupted: {path}")
            print(f"Error: {e}\n")
            bad_files.append(path)

print("=" * 50)
print(f"Total Corrupted Files: {len(bad_files)}")