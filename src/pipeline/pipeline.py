"""
pipeline.py

NeuroShield-ID Integrated Pipeline
"""

import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from src.deepfake_detection.inference import predict, DECISION_THRESHOLD
from src.deepfake_detection.video_processor import (
    extract_video_frames,
    select_representative_frame,
    run_frame_based_video_deepfake_detection,
)
from src.face_verification.verify import verify_faces
from src.trust_score.trust_score import calculate_trust_score
from src.zkp.generate_proof import generate_proof
from src.zkp.verify_proof import verify_proof
from src.blockchain.store_proof import store_proof
from src.explainability.gradcam import generate_gradcam
from src.database.database import save_verification

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

RUN_ROOT = Path("temp/runs")

# The ZKP step reads/writes fixed files in circuits/ (input.json,
# proof.json, public.json), and Grad-CAM shares one hooked model. Streamlit
# serves sessions from threads, so without these locks two simultaneous
# verifications could prove/store each other's data.
_ZKP_LOCK = threading.Lock()
_GRADCAM_LOCK = threading.Lock()


def _prune_old_runs(max_age_seconds=3600):
    """Delete per-run temp folders (frames, Grad-CAM) older than an hour."""

    if not RUN_ROOT.exists():
        return

    now = time.time()

    for d in RUN_ROOT.iterdir():
        try:
            if d.is_dir() and now - d.stat().st_mtime > max_age_seconds:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def _real_probability(deepfake_result):
    """
    Return the probability (0-100) that the input is REAL.

    The new inference pipeline explicitly provides fake_probability and
    real_probability.  Fall back to the old confidence-based heuristic
    for backward compatibility with any callers that use the old format.
    """

    # New pipeline: real_probability is directly available (0-1 scale)
    if "real_probability" in deepfake_result:
        return float(deepfake_result["real_probability"]) * 100.0

    # Video aggregation path: avg_fake_probability is 0-100
    if "avg_fake_probability" in deepfake_result:
        return 100.0 - float(deepfake_result["avg_fake_probability"])

    # Legacy fallback
    confidence = float(deepfake_result["confidence"])
    if str(deepfake_result["prediction"]).upper() == "REAL":
        return confidence
    return 100.0 - confidence


def run_pipeline(reference_image, verification_input, is_video=False):
    """
    Run the complete NeuroShield-ID pipeline.
    Supports both single image and video inputs.
    """

    print("\n===================================")
    print("        NeuroShield-ID")
    print("===================================")

    # Auto-detect video if not explicitly flagged
    input_suffix = Path(verification_input).suffix.lower()
    if is_video or input_suffix in VIDEO_EXTENSIONS:
        is_video = True

    _prune_old_runs()

    # Unique folder per run: frames and the Grad-CAM image no longer
    # collide between sessions or leak in from earlier runs.
    run_dir = RUN_ROOT / uuid.uuid4().hex[:12]
    run_dir.mkdir(parents=True, exist_ok=True)

    extracted_frames = []
    representative_frame = None

    if is_video:
        print("\n--- Video Verification Flow Selected ---")
        # ---------------------------------------
        # Module 1 - Video Frame Extraction & Deepfake Detection
        # ---------------------------------------
        temp_frames_dir = run_dir / "frames"
        extracted_frames = extract_video_frames(
            verification_input,
            temp_frames_dir,
            target_count=5,
        )

        deepfake_result = run_frame_based_video_deepfake_detection(extracted_frames)

        # Representative frame for face verification & Grad-CAM
        representative_frame = select_representative_frame(extracted_frames)
        verification_target_image = representative_frame

    else:
        # ---------------------------------------
        # Module 1 - Deepfake Detection (Single Image)
        # ---------------------------------------
        deepfake_result = predict(verification_input)
        verification_target_image = verification_input

    # ---------------------------------------
    # Module 5 - Explainable AI (Grad-CAM)
    # Run before anything is written to the database/ledger, so a
    # Grad-CAM failure can't leave a stored record for a run that errors.
    # ---------------------------------------
    gradcam_path = str(run_dir / "gradcam_result.jpg")

    with _GRADCAM_LOCK:
        try:
            generate_gradcam(
                verification_target_image,
                gradcam_path,
            )
        except Exception as e:
            print(f"Grad-CAM generation failed (non-fatal): {e}")
            gradcam_path = None

    # ---------------------------------------
    # Module 2 - Face Verification
    # ---------------------------------------
    try:
        verification_result = verify_faces(
            reference_image,
            verification_target_image,
        )
    except Exception as e:
        print(f"Face verification error: {e}")
        verification_result = {
            "verified": False,
            "distance": None,
            "threshold": None,
            "similarity_score": 0.0,
            "error": f"Face verification failed: {str(e)}",
        }

    # ---------------------------------------
    # Module 3 - Trust Score
    # ---------------------------------------
    trust_score, status = calculate_trust_score(
        _real_probability(deepfake_result),
        verification_result["similarity_score"],
    )

    # Identity must actually match to be trusted.
    if verification_result.get("error"):
        status = "REJECTED"
    elif status == "TRUSTED" and not verification_result["verified"]:
        status = "SUSPICIOUS"

    # ---------------------------------------
    # Module 4 - Zero Knowledge Proof
    # ---------------------------------------
    proof_hash = None

    with _ZKP_LOCK:

        proof_generated = generate_proof(trust_score)

        zkp_result = {
            "proof_verified": False,
            "policy_passed": False,
        }

        if proof_generated:
            zkp_result = verify_proof()

        # Only record on the ledger when the proof verifies AND the policy
        # passed.
        if zkp_result["proof_verified"] and zkp_result["policy_passed"]:
            proof_hash = store_proof()

    save_verification(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        prediction=deepfake_result["prediction"],
        confidence=deepfake_result["confidence"],
        similarity=verification_result["similarity_score"],
        trust_score=trust_score,
        status=status,
        proof_hash=proof_hash,
    )

    # ---------------------------------------
    # Final Results Logging
    # ---------------------------------------
    print("\n========== FINAL RESULT ==========\n")
    print(f"Deepfake Prediction : {deepfake_result['prediction']}")
    print(f"REAL Probability    : {deepfake_result.get('real_probability', 0.0) * 100:.2f}%")
    print(f"FAKE Probability    : {deepfake_result.get('fake_probability', 0.0) * 100:.2f}%")
    print(f"Confidence State    : {deepfake_result.get('confidence_state', 'N/A')}")
    print(f"Decision Threshold  : {DECISION_THRESHOLD}")

    if is_video:
        print(f"Video Avg Fake Prob : {deepfake_result.get('avg_fake_probability', 0.0):.2f}%")
        print(f"Video Max Fake Prob : {deepfake_result.get('max_fake_probability', 0.0):.2f}%")
        print(f"Valid Face Frames   : {deepfake_result.get('valid_face_frames', 0)}")
        print(f"Representative Frame: {representative_frame}")

    print()
    print(f"Face Verified       : {verification_result['verified']}")
    print(f"Similarity Score    : {verification_result['similarity_score']:.2f}%")

    print()
    print(f"Trust Score         : {trust_score:.2f}")
    print(f"Verification Status : {status}")

    print()
    print("========== ZERO KNOWLEDGE PROOF ==========\n")
    print(f"Proof Generated : {proof_generated}")
    print(f"Proof Verified  : {zkp_result['proof_verified']}")
    print(f"Policy Passed   : {zkp_result['policy_passed']}")

    return {
        "deepfake": deepfake_result,
        "verification": verification_result,
        "trust_score": trust_score,
        "status": status,
        "decision_threshold": DECISION_THRESHOLD,
        "zkp": {
            "generated": proof_generated,
            "verified": zkp_result["proof_verified"],
            "policy_passed": zkp_result["policy_passed"],
        },
        "blockchain": {
            "proof_hash": proof_hash,
        },
        "gradcam": {
            "image": gradcam_path,
        },
        "is_video": is_video,
        "representative_frame": representative_frame,
        "extracted_frames": extracted_frames,
    }