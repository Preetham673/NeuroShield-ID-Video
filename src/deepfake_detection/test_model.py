from src.deepfake_detection.model import build_model
from src.deepfake_detection.model import DEVICE

model = build_model()

model = model.to(DEVICE)

print()

print(model)

print()

print("Running on:", DEVICE)