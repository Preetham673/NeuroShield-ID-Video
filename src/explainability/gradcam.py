"""
gradcam.py

Generate Grad-CAM visualization for
the trained Deepfake Detection model.
"""

import cv2
import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
)

from src.deepfake_detection.model import (
    build_model,
    DEVICE,
    MODEL_PATH,
)

from src.deepfake_detection.transforms import (
    test_transform,
)

# -----------------------------------
# Load trained model
# -----------------------------------

model = build_model(pretrained=False)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
)

model.to(DEVICE)

model.eval()

# Last convolution layer
target_layers = [model.conv_head]

cam = GradCAM(
    model=model,
    target_layers=target_layers,
)

def generate_gradcam(image_path, output_path="gradcam_result.jpg"):

    image = Image.open(image_path).convert("RGB")

    rgb_image = np.array(image).astype(np.float32) / 255.0

    input_tensor = test_transform(image)

    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    # The CAM is 224x224 (model input size) but rgb_image is the original
    # resolution. show_cam_on_image needs identical shapes, so resize the
    # CAM up to the original image. cv2.resize takes (width, height).
    grayscale_cam = cv2.resize(
        grayscale_cam,
        (rgb_image.shape[1], rgb_image.shape[0]),
    )

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True,
    )

    visualization = cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR,
    )

    cv2.imwrite(
        output_path,
        visualization,
    )

    print()

    print("Grad-CAM saved as:")

    print(output_path)

    return output_path