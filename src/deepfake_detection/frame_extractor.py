from pathlib import Path
import cv2
from tqdm import tqdm

from src.config import FACEFORENSICS_PATH

# =====================================================
# Dataset Paths
# =====================================================

REAL_VIDEO_DIR = FACEFORENSICS_PATH / "FaceForensics++_C23" / "original"

FAKE_VIDEO_DIR = FACEFORENSICS_PATH / "FaceForensics++_C23" / "Deepfakes"

OUTPUT_DIR = Path(r"D:\NeuroShield_Research\processed_data")

REAL_OUTPUT = OUTPUT_DIR / "train" / "real"

FAKE_OUTPUT = OUTPUT_DIR / "train" / "fake"

# =====================================================
# Parameters
# =====================================================

FRAME_SKIP = 10

IMAGE_SIZE = (224, 224)

# =====================================================
# Frame Extraction Function
# =====================================================

def extract_frames(video_path, output_folder, label):
    cap = cv2.VideoCapture(str(video_path))

    frame_number = 0
    saved_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_number % FRAME_SKIP == 0:

            frame = cv2.resize(frame, IMAGE_SIZE)

            filename = f"{video_path.stem}_{saved_count}.jpg"

            save_path = output_folder / filename

            cv2.imwrite(str(save_path), frame)

            saved_count += 1

        frame_number += 1

    cap.release()

    return saved_count

# =====================================================
# Main Function
# =====================================================

def main():

    real_videos = list(REAL_VIDEO_DIR.glob("*.mp4"))
    fake_videos = list(FAKE_VIDEO_DIR.glob("*.mp4"))

    print(f"Found {len(real_videos)} real videos")
    print(f"Found {len(fake_videos)} fake videos")

    print("\nExtracting REAL videos...\n")

    for video in tqdm(real_videos):
        extract_frames(video, REAL_OUTPUT, "real")

    print("\nExtracting FAKE videos...\n")

    for video in tqdm(fake_videos):
        extract_frames(video, FAKE_OUTPUT, "fake")

    print("\nFrame extraction completed successfully!")


if __name__ == "__main__":
    main()