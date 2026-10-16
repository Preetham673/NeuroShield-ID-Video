"""
Global Configuration
NeuroShield-ID
"""

import os
from pathlib import Path

# <root>/src/config.py -> <root>
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------
# Datasets
# Other files import these (frame_extractor.py needs FACEFORENSICS_PATH,
# test_dataset.py needs DATASET_ROOT, test_config.py needs all of them),
# but the config you sent did not define them, so those imports failed.
#
# The defaults below are GUESSES based on your D:\NeuroShield_Research
# folder. Either edit them or set the environment variables.
# --------------------------------------------------

DATASET_ROOT = Path(
    os.environ.get("NEUROSHIELD_DATASET_ROOT", r"D:\NeuroShield_Research")
)

FACEFORENSICS_PATH = Path(
    os.environ.get("NEUROSHIELD_FACEFORENSICS", DATASET_ROOT / "FaceForensics")
)

CELEB_DF_PATH = Path(
    os.environ.get("NEUROSHIELD_CELEB_DF", DATASET_ROOT / "Celeb-DF")
)

LFW_PATH = Path(
    os.environ.get("NEUROSHIELD_LFW", DATASET_ROOT / "LFW")
)

# --------------------------------------------------
# Project paths (absolute, so they no longer depend on the folder the
# app is launched from). Kept as strings like before.
# --------------------------------------------------

MODEL_PATH = str(PROJECT_ROOT / "models" / "deepfake_detector.pth")

OUTPUT_DIR = str(PROJECT_ROOT / "outputs")

GRADCAM_DIR = str(PROJECT_ROOT / "outputs" / "gradcam")

ENCRYPTED_DIR = str(PROJECT_ROOT / "outputs" / "encrypted")

BLOCKCHAIN_DIR = str(PROJECT_ROOT / "outputs" / "blockchain")

LOG_DIR = str(PROJECT_ROOT / "outputs" / "logs")

IMAGE_SIZE = 224

# --------------------------------------------------
# Trust score policy (used by trust_score.py, so the numbers live in
# one place instead of being hardcoded there and here).
# NOTE: if your Circom circuit hardcodes its own threshold, keep it equal
# to TRUST_THRESHOLD or the ZKP policy result will disagree with the status.
# --------------------------------------------------

TRUST_THRESHOLD = 75

SUSPICIOUS_THRESHOLD = 50

DEEPFAKE_WEIGHT = 0.6

FACE_WEIGHT = 0.4