"""
inference.py

Predict whether an image is Real or Fake using the trained EfficientNet-B0.

Key design decisions
--------------------
* REAL = class 0, FAKE = class 1 (matches training labels).
* The model outputs raw logits; softmax is applied here.
* fake_probability  = softmax(logits)[1]   (probability of class FAKE)
* real_probability  = softmax(logits)[0]   (probability of class REAL)
* Decision threshold: FAKE when fake_probability >= DECISION_THRESHOLD
  (default 0.50 - not inflated to hide false positives).
* Confidence state is based on distance from the threshold, NOT model
  accuracy, so it is clearly labelled "prediction confidence" not
  "calibrated probability".

Public API
----------
preprocess_for_inference(pil_image)  -> torch.Tensor (C,H,W)
detect_face_crop(pil_image)          -> PIL.Image | None
predict_single_image(pil_image)      -> dict
predict_multi_crop(pil_image)        -> dict
aggregate_predictions(probs)         -> float   (mean fake_prob)
calculate_confidence_state(fake_prob)-> str
predict(image_path)                  -> dict    (legacy path-based wrapper)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.deepfake_detection.model import build_model, DEVICE, MODEL_PATH
from src.deepfake_detection.transforms import test_transform

logger = logging.getLogger(__name__)

# ------------------------------------
# Constants
# ------------------------------------

CLASSES = ["REAL", "FAKE"]

# Decision threshold: fake_probability >= DECISION_THRESHOLD -> FAKE.
# Exposed as a module-level constant so callers can inspect/override it.
DECISION_THRESHOLD: float = 0.50

# Confidence-state boundaries (distance from threshold).
# BORDERLINE if the fake_probability is within BORDERLINE_MARGIN of threshold.
BORDERLINE_MARGIN: float = 0.15

# Minimum face crop dimension in pixels.
MIN_FACE_DIM: int = 60

# ------------------------------------
# Load Trained Model (once at import)
# ------------------------------------

_model = build_model(pretrained=False)

_model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
)

_model.to(DEVICE)
_model.eval()

# ============================================================
# Core preprocessing helpers
# ============================================================


def preprocess_for_inference(pil_image: Image.Image) -> torch.Tensor:
    """
    Apply the exact same transform used at validation/test time.

    Parameters
    ----------
    pil_image : PIL.Image
        Input image.  Must already be in RGB mode.

    Returns
    -------
    torch.Tensor shape (C, H, W)  - NOT batched.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return test_transform(pil_image)


def detect_face_crop(
    pil_image: Image.Image,
    margin: float = 0.25,
) -> Optional[Image.Image]:
    """
    Detect the largest face in pil_image and return a cropped region.

    The crop includes a configurable margin around the detected bounding
    box so that manipulation boundaries (hair-line, chin, ears) are not
    lost.  Returns None when no face is detected.

    Parameters
    ----------
    pil_image : PIL.Image (RGB)
    margin    : fraction of face size added as padding on all sides
                (0.25 = 25% of face width/height).

    Returns
    -------
    PIL.Image of the face crop, or None.
    """
    import cv2

    # Convert PIL RGB -> OpenCV BGR
    cv_img = np.array(pil_image.convert("RGB"))[:, :, ::-1].copy()
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = []
    try:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(MIN_FACE_DIM, MIN_FACE_DIM),
        )
    except Exception as exc:
        logger.debug("Haar cascade failed: %s", exc)

    if not isinstance(faces, np.ndarray) or len(faces) == 0:
        return None

    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

    ih, iw = cv_img.shape[:2]
    pad_x = int(w * margin)
    pad_y = int(h * margin)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(iw, x + w + pad_x)
    y2 = min(ih, y + h + pad_y)

    # Guard against degenerate crop
    if (x2 - x1) < MIN_FACE_DIM or (y2 - y1) < MIN_FACE_DIM:
        return None

    crop = pil_image.crop((x1, y1, x2, y2))
    return crop


# ============================================================
# Single-image prediction
# ============================================================


def predict_single_image(pil_image: Image.Image) -> dict:
    """
    Run the EfficientNet-B0 model on a single PIL image.

    Returns
    -------
    dict with keys:
        fake_probability  (float, 0-1)
        real_probability  (float, 0-1)
        prediction        ("REAL" | "FAKE")
        confidence        (float, 0-100)  - confidence in predicted class
        threshold         (float)
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    tensor = preprocess_for_inference(pil_image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    real_prob = float(probs[0].item())
    fake_prob = float(probs[1].item())

    prediction = "FAKE" if fake_prob >= DECISION_THRESHOLD else "REAL"
    confidence = fake_prob * 100.0 if prediction == "FAKE" else real_prob * 100.0

    return {
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "prediction": prediction,
        "confidence": confidence,
        "threshold": DECISION_THRESHOLD,
    }


# ============================================================
# Multi-crop aggregation
# ============================================================


def predict_multi_crop(pil_image: Image.Image) -> dict:
    """
    Robust inference using multiple views of the same image.

    Views used
    ----------
    1. Standard full-image inference (test_transform -> 224x224).
    2. Face-centred crop with 25% margin (if a face is detected).
    3. Face-centred crop with a tighter 10% margin (if a face is detected).

    Aggregation
    -----------
    Mean FAKE probability across all successful crops.
    The final prediction is: FAKE if mean_fake_prob >= DECISION_THRESHOLD.

    This is a transparent, model-agnostic aggregation that reduces
    single-crop variance without inventing a new model.

    Returns
    -------
    dict with all single-image keys plus:
        crops_used        (int)
        crop_fake_probs   (list of float)
        crop_labels       (list of str)
        mean_fake_prob    (float, 0-1)
        confidence_state  (str)
        method            str
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    crop_probs: list = []
    crop_labels: list = []

    # --- View 1: full image ---
    r1 = predict_single_image(pil_image)
    crop_probs.append(r1["fake_probability"])
    crop_labels.append("full_image")

    # --- Views 2 & 3: face crops ---
    for margin, label in [(0.25, "face_crop_25pct"), (0.10, "face_crop_10pct")]:
        face_crop = detect_face_crop(pil_image, margin=margin)
        if face_crop is not None:
            r = predict_single_image(face_crop)
            crop_probs.append(r["fake_probability"])
            crop_labels.append(label)

    mean_fake_prob = float(np.mean(crop_probs))
    mean_real_prob = 1.0 - mean_fake_prob

    prediction = "FAKE" if mean_fake_prob >= DECISION_THRESHOLD else "REAL"
    confidence = mean_fake_prob * 100.0 if prediction == "FAKE" else mean_real_prob * 100.0
    confidence_state = calculate_confidence_state(mean_fake_prob)

    return {
        "fake_probability": mean_fake_prob,
        "real_probability": mean_real_prob,
        "prediction": prediction,
        "confidence": confidence,
        "confidence_state": confidence_state,
        "threshold": DECISION_THRESHOLD,
        "crops_used": len(crop_probs),
        "crop_fake_probs": crop_probs,
        "crop_labels": crop_labels,
        "method": "multi_crop_mean",
    }


# ============================================================
# Aggregation helper
# ============================================================


def aggregate_predictions(fake_probs: list) -> float:
    """
    Aggregate a list of per-crop FAKE probabilities into a single value.

    Currently uses mean aggregation for transparency.
    """
    if not fake_probs:
        raise ValueError("aggregate_predictions: empty probability list")
    return float(np.mean(fake_probs))


# ============================================================
# Confidence state
# ============================================================


def calculate_confidence_state(fake_prob: float) -> str:
    """
    Return a human-readable confidence state based on how far the
    fake_probability is from the decision threshold.

    This is NOT a calibrated probability - it is a descriptive label
    for the distance from the threshold.

    States
    ------
    HIGH CONFIDENCE  : |fake_prob - threshold| > BORDERLINE_MARGIN
    BORDERLINE       : |fake_prob - threshold| <= BORDERLINE_MARGIN
    """
    distance = abs(fake_prob - DECISION_THRESHOLD)
    if distance <= BORDERLINE_MARGIN:
        return "BORDERLINE"
    return "HIGH CONFIDENCE"


# ============================================================
# Legacy path-based wrapper  (used by video_processor, pipeline, etc.)
# ============================================================


def predict(image_path) -> dict:
    """
    Predict whether an image file is REAL or FAKE.

    This is the legacy entry-point used by video_processor.py,
    pipeline.py and gradcam.py.  It now delegates to predict_multi_crop()
    so all path-based callers automatically get the improved pipeline.

    Parameters
    ----------
    image_path : str or Path
        Path to an image file.  PIL handles JPEGs saved by OpenCV
        (which are already in BGR-on-disk, but PIL reads them as RGB,
        matching the training pipeline).

    Returns
    -------
    dict:
        prediction        ("REAL" | "FAKE")
        confidence        (float, 0-100)
        fake_probability  (float, 0-1)
        real_probability  (float, 0-1)
        confidence_state  (str)
        threshold         (float)
        crops_used        (int)
        crop_fake_probs   (list[float])
        crop_labels       (list[str])
        input_size        (tuple)
        processed_size    (tuple)
        method            (str)
    """
    if isinstance(image_path, Image.Image):
        pil_image = image_path.convert("RGB")
    else:
        pil_image = Image.open(image_path).convert("RGB")
    input_size = pil_image.size  # (width, height)

    result = predict_multi_crop(pil_image)
    result["input_size"] = input_size
    result["processed_size"] = (224, 224)  # model input size

    # Print summary for console logging
    print("\n========== DEEPFAKE DETECTION ==========\n")
    print(f"Image           : {image_path}")
    print(f"Prediction      : {result['prediction']}")
    print(f"REAL Probability: {result['real_probability'] * 100:.2f}%")
    print(f"FAKE Probability: {result['fake_probability'] * 100:.2f}%")
    print(f"Confidence      : {result['confidence']:.2f}%  [{result['confidence_state']}]")
    print(f"Threshold       : {DECISION_THRESHOLD}")
    print(f"Crops used      : {result['crops_used']}")
    for lbl, p in zip(result["crop_labels"], result["crop_fake_probs"]):
        print(f"  {lbl:<22}: FAKE {p * 100:.2f}%")

    return result