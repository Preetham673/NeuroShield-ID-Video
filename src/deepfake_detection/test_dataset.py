from src.config import DATASET_ROOT
from src.deepfake_detection.dataset import DeepfakeDataset
from src.deepfake_detection.transforms import train_transform

train_path = r"D:\NeuroShield_Research\processed_data\train"

dataset = DeepfakeDataset(
    train_path,
    transform=train_transform
)

print()

print("Dataset Size:", len(dataset))

image, label = dataset[0]

print()

print("Image Shape:", image.shape)

print("Label:", label)