"""
video_processor.py

Frame extraction, face detection, frame-level deepfake inference, and
representative frame selection for video verification in NeuroShield-ID.

Aggregation strategy
--------------------
1. Extract target_count representative frames distributed 10%-90% through
   the video to avoid black/transition edges.
2. For each frame attempt to detect a face.  Frames WITH a detected face
   are used preferentially; frames without a face are used as fallback if
   there are not enough face frames.
3. Run the EfficientNet-B0 deepfake detector on each valid frame using the
   multi-crop inference pipeline (full image + face crops).
4. Video-level FAKE probability = mean of per-frame FAKE probabilities.
5. A video is classified as FAKE only when the MEAN fake probability
   >= DECISION_THRESHOLD (currently 0.50).  A single anomalous frame is
   insufficient to trigger a FAKE verdict.
6. Additionally report:
   - total_frames_extracted
   - valid_face_frames
   - avg_fake_probability
   - max_fake_probability
   - confidence_state of the video-level decision
"""

import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

from src.deepfake_detection.inference import (
    predict,
    DECISION_THRESHOLD,
    calculate_confidence_state,
    detect_face_crop,
)


# ---------------------------------------------------------------------------
# Fallback Video Decoder & Helper Functions
# ---------------------------------------------------------------------------


def get_ffmpeg_executable() -> str:
    """Return path to ffmpeg executable (via imageio-ffmpeg or system PATH)."""
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    # Fallback to system ffmpeg if installed
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    return ""


def convert_video_with_ffmpeg(src_path: str, dst_path: str, codec: str = "mjpeg") -> bool:
    """
    Convert video file using FFmpeg into a format that OpenCV on Windows
    can decode reliably without requiring external H.264 videoio dlls.
    """
    ffmpeg_exe = get_ffmpeg_executable()
    if not ffmpeg_exe:
        return False

    cmd = [ffmpeg_exe, "-y", "-i", str(src_path)]
    if codec == "mjpeg":
        cmd += ["-c:v", "mjpeg", "-q:v", "2"]
    elif codec == "h264":
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    cmd.append(str(dst_path))

    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        return res.returncode == 0 and os.path.exists(dst_path) and os.path.getsize(dst_path) > 0
    except Exception as e:
        print(f"[Video Decoder] FFmpeg conversion failed: {e}")
        return False


def open_video_with_fallback(video_path: str):
    """
    Robust video opener with automatic fallback:
    1. Verify file exists and has size > 0.
    2. Attempt direct cv2.VideoCapture.
    3. If OpenCV fails to open or cannot read frames:
       a. Convert to a temporary standard MP4 / AVI (MJPEG) using FFmpeg.
       b. Attempt to open converted video with OpenCV.
    4. Returns tuple: (cap: cv2.VideoCapture | None, cleanup_path: str | None, debug_info: dict)
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    file_size = video_path_obj.stat().st_size
    if file_size == 0:
        raise ValueError("Video file is empty (0 bytes).")

    ext = video_path_obj.suffix.lower()
    debug_info = {
        "original_filename": video_path_obj.name,
        "file_extension": ext,
        "file_size_bytes": file_size,
        "opencv_direct_open": False,
        "fallback_conversion_used": False,
        "converted_file": None,
        "num_frames": 0,
        "fps": 0.0,
        "duration_sec": 0.0,
    }

    # Step A: Try direct OpenCV open
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Verify it can actually read at least one frame
        ret, frame = cap.read()
        if ret and frame is not None:
            # Rewind
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            debug_info["opencv_direct_open"] = True
            debug_info["num_frames"] = total_frames
            debug_info["fps"] = fps
            debug_info["duration_sec"] = round(total_frames / fps, 2) if fps > 0 and total_frames > 0 else 0.0
            print(f"[Video Decoder] Direct OpenCV open successful for {video_path_obj.name}")
            return cap, None, debug_info
        else:
            cap.release()

    # Step B: Direct OpenCV failed or could not decode frames -> Use FFmpeg fallback
    print(f"[Video Decoder] Direct OpenCV open failed for {video_path_obj.name}. Attempting FFmpeg fallback...")

    # We try converting to temporary standard MP4 and AVI (MJPEG)
    temp_dir = video_path_obj.parent
    stem = video_path_obj.stem
    converted_path = temp_dir / f"{stem}_converted.avi"

    conversion_ok = convert_video_with_ffmpeg(str(video_path), str(converted_path), codec="mjpeg")
    if not conversion_ok:
        # Try MP4 as alternative
        converted_path = temp_dir / f"{stem}_converted.mp4"
        conversion_ok = convert_video_with_ffmpeg(str(video_path), str(converted_path), codec="h264")

    if conversion_ok and converted_path.exists():
        cap_conv = cv2.VideoCapture(str(converted_path))
        if cap_conv.isOpened():
            ret, frame = cap_conv.read()
            if ret and frame is not None:
                cap_conv.set(cv2.CAP_PROP_POS_FRAMES, 0)
                total_frames = int(cap_conv.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap_conv.get(cv2.CAP_PROP_FPS) or 25.0)
                debug_info["fallback_conversion_used"] = True
                debug_info["converted_file"] = str(converted_path)
                debug_info["num_frames"] = total_frames
                debug_info["fps"] = fps
                debug_info["duration_sec"] = round(total_frames / fps, 2) if fps > 0 and total_frames > 0 else 0.0
                print(f"[Video Decoder] FFmpeg fallback conversion successful: {converted_path}")
                return cap_conv, str(converted_path), debug_info
            cap_conv.release()

    # If both direct OpenCV and OpenCV on converted file fail:
    if converted_path.exists():
        try:
            os.remove(converted_path)
        except Exception:
            pass

    return None, None, debug_info


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_video_frames(video_path: str, output_dir: Path, target_count: int = 5):
    """
    Extract approximately target_count representative frames distributed
    across the video using open_video_with_fallback, with direct FFmpeg
    extraction fallback if needed.

    Parameters
    ----------
    video_path : str
        Path to the temporary video file.
    output_dir : Path
        Directory where extracted frames will be saved.
    target_count : int
        Target number of frames to extract (default 5).

    Returns
    -------
    list of dict
        Extracted frame info:
        [{"frame_number": int, "path": str, "timestamp_sec": float}]
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path_obj = Path(video_path)

    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cap, cleanup_video, debug_info = open_video_with_fallback(video_path)
    saved_frames = []

    try:
        if cap is not None and cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

            # Handle cases where total_frames is not reported properly
            if total_frames <= 0:
                frames_list = []
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    frames_list.append((frame_idx, frame))
                    frame_idx += 1
                    if len(frames_list) >= 300:
                        break

                if frames_list:
                    step = max(1, len(frames_list) // target_count)
                    selected_frames = frames_list[::step][:target_count]
                    for f_num, frame in selected_frames:
                        frame_path = output_dir / f"frame_{f_num}.jpg"
                        cv2.imwrite(str(frame_path), frame)
                        saved_frames.append(
                            {
                                "frame_number": f_num,
                                "path": str(frame_path),
                                "timestamp_sec": round(f_num / fps, 2),
                            }
                        )
            else:
                # Sample evenly across 10% to 90% of the video to avoid black edges
                count = min(target_count, total_frames)
                if count == 1:
                    indices = [total_frames // 2]
                else:
                    indices = np.linspace(
                        int(total_frames * 0.1),
                        int(total_frames * 0.9),
                        num=count,
                        dtype=int,
                    )
                    indices = [int(idx) for idx in indices]

                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        ret, frame = cap.read()

                    if ret and frame is not None:
                        frame_path = output_dir / f"frame_{idx}.jpg"
                        cv2.imwrite(str(frame_path), frame)
                        saved_frames.append(
                            {
                                "frame_number": idx,
                                "path": str(frame_path),
                                "timestamp_sec": round(idx / fps, 2),
                            }
                        )
        else:
            # If OpenCV could not decode even after conversion, try FFmpeg direct extraction
            ffmpeg_exe = get_ffmpeg_executable()
            if ffmpeg_exe:
                print(f"[Video Decoder] Attempting direct FFmpeg frame extraction for {video_path}")
                temp_extract_dir = output_dir / "raw_extract"
                temp_extract_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-i",
                    str(video_path),
                    "-vf",
                    "fps=1",
                    str(temp_extract_dir / "frame_%04d.jpg"),
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
                raw_files = sorted(temp_extract_dir.glob("frame_*.jpg"))
                if raw_files:
                    step = max(1, len(raw_files) // target_count)
                    selected_files = raw_files[::step][:target_count]
                    for idx, src_f in enumerate(selected_files):
                        target_f = output_dir / f"frame_{idx}.jpg"
                        shutil.copy2(src_f, target_f)
                        saved_frames.append(
                            {
                                "frame_number": idx,
                                "path": str(target_f),
                                "timestamp_sec": round(float(idx), 2),
                            }
                        )
                    debug_info["fallback_conversion_used"] = True
                shutil.rmtree(temp_extract_dir, ignore_errors=True)

    finally:
        if cap is not None:
            cap.release()
        if cleanup_video and os.path.exists(cleanup_video):
            try:
                os.remove(cleanup_video)
            except Exception:
                pass

    if not saved_frames:
        raise ValueError(
            "OpenCV could not decode this video directly. NeuroShield-ID attempted "
            "automatic format conversion, but the video codec could not be decoded."
        )

    # Attach decoder debug info so pipeline and UI can inspect it
    for f in saved_frames:
        f["decoder_debug"] = debug_info

    return saved_frames


# ---------------------------------------------------------------------------
# Representative frame selection
# ---------------------------------------------------------------------------


def select_representative_frame(extracted_frames: list) -> str:
    """
    Select the frame with the best detectable face / clearest frame for
    face verification and Grad-CAM.

    Scoring: frames with a detected face score >= 1_000_000; within those,
    larger faces and sharper images score higher.

    Parameters
    ----------
    extracted_frames : list of dict

    Returns
    -------
    str
        File path of the selected representative frame.
    """
    if not extracted_frames:
        raise ValueError(
            "No frames available to select representative verification image."
        )

    face_cascade = None
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
    except Exception:
        face_cascade = None

    best_frame_path = extracted_frames[0]["path"]
    best_score = -1.0

    for item in extracted_frames:
        f_path = item["path"]
        img = cv2.imread(f_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clarity = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        face_detected = False
        face_area = 0

        if face_cascade is not None and not face_cascade.empty():
            try:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30),
                )
                if len(faces) > 0:
                    face_detected = True
                    face_area = max(w * h for (x, y, w, h) in faces)
            except Exception:
                face_detected = False

        # Prioritise frames with detected faces, break ties by size + sharpness
        if face_detected:
            score = 1_000_000.0 + face_area + min(clarity, 1000.0)
        else:
            score = clarity

        if score > best_score:
            best_score = score
            best_frame_path = f_path

    return best_frame_path


# ---------------------------------------------------------------------------
# Video-level deepfake detection
# ---------------------------------------------------------------------------


def run_frame_based_video_deepfake_detection(extracted_frames: list) -> dict:
    """
    Run the EfficientNet-B0 deepfake detector on each extracted frame and
    aggregate results into a transparent video-level classification.

    Face detection is attempted for each frame.  Frames that contain a
    detectable face are flagged so the result dict can report how many
    valid face frames were available.

    Aggregation rule
    ----------------
    * Compute the mean FAKE probability across all analysed frames.
    * Classify as FAKE only if mean >= DECISION_THRESHOLD (0.50).
    * A single high-FAKE frame is NOT sufficient to call the entire video
      fake — the mean must cross the threshold.
    * Additionally report max_fake_probability so borderline cases are
      visible to the user.

    Returns
    -------
    dict
        {
            "prediction"            : "REAL" | "FAKE",
            "confidence"            : float (0-100),
            "avg_fake_probability"  : float (0-100),
            "max_fake_probability"  : float (0-100),
            "confidence_state"      : str,
            "method"                : str,
            "total_frames_analyzed" : int,
            "valid_face_frames"     : int,
            "frames"                : list[dict],
        }
    """
    if not extracted_frames:
        raise ValueError("No extracted frames to analyze.")

    # Set up Haar cascade for face detection
    face_cascade = None
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
    except Exception:
        pass

    frame_results = []
    fake_probs = []
    valid_face_count = 0

    for item in extracted_frames:
        f_path = item["path"]
        f_num = item["frame_number"]
        timestamp = item.get("timestamp_sec", 0.0)

        # Check if this frame contains a face
        face_found = False
        if face_cascade is not None and not face_cascade.empty():
            try:
                img = cv2.imread(f_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                    )
                    face_found = isinstance(faces, np.ndarray) and len(faces) > 0
            except Exception:
                face_found = False

        if face_found:
            valid_face_count += 1

        # Run multi-crop deepfake inference on this frame
        pred = predict(f_path)
        fake_prob = float(pred["fake_probability"])  # already 0-1
        fake_probs.append(fake_prob)

        frame_results.append(
            {
                "frame_number": f_num,
                "timestamp_sec": timestamp,
                "prediction": str(pred["prediction"]).upper(),
                "confidence": float(pred["confidence"]),
                "confidence_state": pred.get("confidence_state", ""),
                "fake_probability": round(fake_prob * 100.0, 2),
                "real_probability": round((1.0 - fake_prob) * 100.0, 2),
                "face_detected": face_found,
                "crops_used": pred.get("crops_used", 1),
                "image_path": f_path,
            }
        )

    # --- Aggregation: mean FAKE probability ---
    avg_fake_prob = float(np.mean(fake_probs))
    max_fake_prob = float(np.max(fake_probs))

    if avg_fake_prob >= DECISION_THRESHOLD:
        video_prediction = "FAKE"
        video_confidence = avg_fake_prob * 100.0
    else:
        video_prediction = "REAL"
        video_confidence = (1.0 - avg_fake_prob) * 100.0

    confidence_state = calculate_confidence_state(avg_fake_prob)

    decoder_debug = extracted_frames[0].get("decoder_debug", {}) if extracted_frames else {}

    return {
        "prediction": video_prediction,
        "confidence": video_confidence,
        "avg_fake_probability": round(avg_fake_prob * 100.0, 2),
        "max_fake_probability": round(max_fake_prob * 100.0, 2),
        "confidence_state": confidence_state,
        "method": "Frame-based Video Analysis (mean FAKE probability)",
        "total_frames_analyzed": len(frame_results),
        "valid_face_frames": valid_face_count,
        "decoder_debug": decoder_debug,
        "frames": frame_results,
    }
