"""
test_video_pipeline.py

Comprehensive test suite verifying:
1. Video frame extraction
2. Frame-level deepfake inference & aggregation
3. Representative frame selection
4. Face verification on video
5. Full end-to-end pipeline with video input
6. Full end-to-end pipeline with image input (backward compatibility)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.deepfake_detection.video_processor import (
    extract_video_frames,
    select_representative_frame,
    run_frame_based_video_deepfake_detection,
)
from src.pipeline.pipeline import run_pipeline
from src.reports.pdf_report import generate_pdf


def create_test_video(output_path: str, base_image_path: str, num_frames: int = 30):
    img = cv2.imread(base_image_path)
    if img is None:
        raise ValueError(f"Could not load base image from {base_image_path}")

    h, w, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (w, h))

    for i in range(num_frames):
        # Slightly alter brightness to simulate realistic video variations
        factor = 1.0 + (i % 5 - 2) * 0.02
        frame = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        out.write(frame)

    out.release()
    print(f"Created test video at: {output_path} ({num_frames} frames)")


def run_tests():
    print("========================================")
    print("Testing Video Verification Pipeline")
    print("========================================")

    ref_path = "temp/reference.jpg"
    ver_path = "temp/verification.jpg"
    video_path = "temp/test_verification_video.mp4"

    # Step 1: Create a test video from existing verification face
    create_test_video(video_path, ver_path, num_frames=30)

    # Step 2: Test frame extraction
    frames_dir = Path("temp/test_extracted_frames")
    extracted = extract_video_frames(video_path, frames_dir, target_count=5)
    print(f"Extracted {len(extracted)} representative frames:")
    for f in extracted:
        print(f"  - Frame {f['frame_number']} at {f['timestamp_sec']}s: {f['path']}")
    assert len(extracted) == 5, f"Expected 5 frames, got {len(extracted)}"

    # Step 3: Test frame-level deepfake inference & aggregation
    video_df_result = run_frame_based_video_deepfake_detection(extracted)
    print("\nDeepfake Video Result:")
    print(f"  Prediction: {video_df_result['prediction']}")
    print(f"  Confidence: {video_df_result['confidence']:.2f}%")
    print(f"  Avg Fake Prob: {video_df_result['avg_fake_probability']:.2f}%")
    print(f"  Method: {video_df_result['method']}")
    assert "Frame-based Video Analysis" in video_df_result["method"]
    assert "prediction" in video_df_result
    assert "confidence" in video_df_result
    assert len(video_df_result["frames"]) == 5

    # Step 4: Test representative frame selection
    rep_frame = select_representative_frame(extracted)
    print(f"\nRepresentative Frame Selected: {rep_frame}")
    assert Path(rep_frame).exists()

    # Step 5: Test full video pipeline run
    print("\n--- Running Full Pipeline with Video Input ---")
    video_pipeline_result = run_pipeline(ref_path, video_path, is_video=True)
    assert video_pipeline_result["is_video"] is True
    assert "deepfake" in video_pipeline_result
    assert "verification" in video_pipeline_result
    assert "trust_score" in video_pipeline_result
    assert "status" in video_pipeline_result
    assert "zkp" in video_pipeline_result
    assert "blockchain" in video_pipeline_result
    assert "gradcam" in video_pipeline_result
    print("\nVideo Pipeline Result Summary:")
    print(f"  Status: {video_pipeline_result['status']}")
    print(f"  Trust Score: {video_pipeline_result['trust_score']:.2f}")
    print(f"  Deepfake: {video_pipeline_result['deepfake']['prediction']} ({video_pipeline_result['deepfake']['confidence']:.2f}%)")
    print(f"  Face Similarity: {video_pipeline_result['verification']['similarity_score']:.2f}%")

    # Step 6: Test PDF generation with video result
    pdf_path = generate_pdf(video_pipeline_result)
    assert Path(pdf_path).exists()
    print(f"\nPDF Report generated successfully: {pdf_path}")

    # Step 7: Test full image pipeline run (backward compatibility)
    print("\n--- Running Full Pipeline with Standard Image Input ---")
    image_pipeline_result = run_pipeline(ref_path, ver_path, is_video=False)
    assert image_pipeline_result["is_video"] is False
    assert "trust_score" in image_pipeline_result
    print("\nImage Pipeline Result Summary:")
    print(f"  Status: {image_pipeline_result['status']}")
    print(f"  Trust Score: {image_pipeline_result['trust_score']:.2f}")

    print("\n========================================")
    print("ALL TESTS PASSED SUCCESSFULLY! (100%)")
    print("========================================")


if __name__ == "__main__":
    run_tests()
