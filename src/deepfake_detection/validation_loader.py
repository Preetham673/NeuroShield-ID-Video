from torch.utils.data import DataLoader

from src.deepfake_detection.dataset import DeepfakeDataset
from src.deepfake_detection.transforms import test_transform

VALIDATION_PATH = r"D:\NeuroShield_Research\processed_data_subset\validation"

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