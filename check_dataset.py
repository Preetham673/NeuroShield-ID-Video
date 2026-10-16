from PIL import Image
import os

dataset_path = r"D:\NeuroShield_Research\processed_data_subset\test"

bad_files = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        path = os.path.join(root, file)
        try:
            img = Image.open(path)
            img.verify()  # Verify image integrity
        except Exception as e:
            print(f"Corrupted: {path}")
            print(f"Error: {e}\n")
            bad_files.append(path)

print("=" * 50)
print(f"Total Corrupted Files: {len(bad_files)}")