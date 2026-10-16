import os
import time
import uuid
from pathlib import Path
import streamlit as st

from src.reports.pdf_report import generate_pdf
from src.pipeline.pipeline import run_pipeline

from components.ui import (
    page_header,
    section_divider,
    render_badge,
    circular_gauge,
    labeled_progress,
    code_card,
    skeleton,
    glass_card_open,
    glass_card_close,
)

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Image quality helpers
# ---------------------------------------------------------------------------

def _check_image_quality(pil_image):
    """
    Check basic image quality.  Returns (ok: bool, warnings: list[str]).

    Does NOT treat quality issues as deepfake evidence — only warns the user.
    """
    import numpy as np

    warnings = []

    w, h = pil_image.size
    if w < 100 or h < 100:
        warnings.append(
            f"Image resolution is very low ({w}x{h} px). "
            "A larger, clearer image will improve detection accuracy."
        )

    # Blur detection via Laplacian variance
    import cv2
    cv_img = np.array(pil_image.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 50:
        warnings.append(
            f"Image appears blurry (sharpness score: {lap_var:.1f}). "
            "A sharper image is recommended for reliable detection."
        )

    return len(warnings) == 0, warnings


# ---------------------------------------------------------------------------
# Confidence-state display helper
# ---------------------------------------------------------------------------

def _render_confidence_state(confidence_state: str, fake_prob: float):
    """Display confidence state with appropriate styling."""
    if confidence_state == "BORDERLINE":
        st.warning(
            f"⚠️ **Borderline result** (FAKE probability: {fake_prob * 100:.1f}%) — "
            "additional frames or a clearer image are recommended."
        )
    else:
        if fake_prob >= 0.5:
            st.info(f"🔴 High-confidence FAKE signal (FAKE probability: {fake_prob * 100:.1f}%)")
        else:
            st.info(f"🟢 High-confidence REAL signal (FAKE probability: {fake_prob * 100:.1f}%)")


# ---------------------------------------------------------------------------
# Real Camera Test mode
# ---------------------------------------------------------------------------

def _render_real_camera_test():
    """
    Optional Real Camera Test mode for evaluating false-positive rate
    on real webcam images.

    Does NOT claim improved accuracy — reports raw model output only.
    """
    st.markdown('<div class="ns-eval-card">', unsafe_allow_html=True)
    st.markdown("### 📷 Real Camera Test Mode")
    st.caption(
        "Capture multiple real webcam images to evaluate the deepfake detector's "
        "false-positive rate on genuine camera images.  "
        "This does NOT modify the model — it only measures current performance."
    )

    from src.deepfake_detection.inference import predict_multi_crop, DECISION_THRESHOLD

    num_captures = st.slider(
        "Number of test captures", min_value=1, max_value=5, value=3,
        key="rct_num_captures"
    )

    test_results = []

    for i in range(num_captures):
        st.markdown(f"#### Capture {i + 1}")
        captured = st.camera_input(
            f"Take photo {i + 1} for testing",
            key=f"rct_camera_{i}",
            help="Face the camera directly, use good lighting.",
        )
        if captured is not None:
            from PIL import Image
            pil_img = Image.open(captured).convert("RGB")
            result = predict_multi_crop(pil_img)

            col_img, col_res = st.columns([1, 2])
            with col_img:
                st.image(pil_img, caption=f"Test image {i + 1}", use_container_width=True)
            with col_res:
                badge = "success" if result["prediction"] == "REAL" else "danger"
                render_badge(result["prediction"], badge)
                st.write(f"REAL probability: **{result['real_probability'] * 100:.2f}%**")
                st.write(f"FAKE probability: **{result['fake_probability'] * 100:.2f}%**")
                st.write(f"Decision threshold: **{DECISION_THRESHOLD}**")
                st.write(f"Confidence state: **{result['confidence_state']}**")
                st.write(f"Crops used: **{result['crops_used']}**")
                if result['confidence_state'] == 'BORDERLINE':
                    st.warning("Borderline result — consider retaking with better lighting.")

            test_results.append(result)
            st.markdown("---")

    if test_results:
        total = len(test_results)
        fake_count = sum(1 for r in test_results if r["prediction"] == "FAKE")
        real_count = total - fake_count
        st.markdown(
            f"**Test summary:** {total} image(s) tested — "
            f"**{real_count} REAL**, **{fake_count} FAKE**. "
            f"False-positive rate on this sample: "
            f"**{fake_count / total * 100:.0f}%** "
            f"(ideal = 0% for genuine camera images)."
        )
        st.caption(
            "⚠️ This is a qualitative spot-check only. "
            "A statistically valid evaluation requires a labelled validation dataset."
        )
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():

    page_header(
        "🔍",
        "Identity Verification",
        "Authenticate identity using live camera, uploaded images, or frame-based video analysis.",
    )

    # -------------------- Upload / Capture Cards --------------------

    col1, col2 = st.columns(2)

    with col1:
        glass_card_open()
        st.markdown("#### 📎 REFERENCE IMAGE")
        ref_source = st.radio(
            "Reference Input Mode",
            ["Upload Reference Image", "Take Live Reference Photo"],
            key="ref_source",
            horizontal=True,
        )

        reference_file = None
        if ref_source == "Upload Reference Image":
            reference_file = st.file_uploader(
                "Upload Reference Image",
                type=["jpg", "jpeg", "png"],
                key="reference_uploader",
                help="Upload reference photo (.jpg, .jpeg, .png)",
            )
        else:
            reference_file = st.camera_input(
                "Take Live Reference Photo",
                key="reference_camera",
                help="Allow browser camera permissions to capture photo",
            )

        if reference_file is not None:
            st.image(reference_file, caption="Reference Preview", use_container_width=True)
        glass_card_close()

    with col2:
        glass_card_open()
        st.markdown("#### 📎 VERIFICATION INPUT")
        ver_type = st.radio(
            "Choose input type:",
            ["Image", "Video"],
            key="ver_type",
            horizontal=True,
        )

        verification_file = None
        is_video_input = (ver_type == "Video")

        if not is_video_input:
            img_source = st.radio(
                "Image Source",
                ["Upload Verification Image", "Take Live Verification Photo"],
                key="ver_img_source",
                horizontal=True,
            )
            if img_source == "Upload Verification Image":
                verification_file = st.file_uploader(
                    "Upload Verification Image",
                    type=["jpg", "jpeg", "png"],
                    key="ver_img_uploader",
                    help="Upload verification image (.jpg, .jpeg, .png)",
                )
            else:
                # Camera capture guidance
                st.info(
                    "📸 **Webcam tips for best results:**\n"
                    "- Face the camera directly\n"
                    "- Use good, even lighting\n"
                    "- Avoid harsh shadows\n"
                    "- Keep your face clearly visible in frame\n"
                    "- Stay still while capturing"
                )
                verification_file = st.camera_input(
                    "Take Live Verification Photo",
                    key="ver_camera_input",
                )

            if verification_file is not None:
                st.image(
                    verification_file,
                    caption="Verification Image Preview",
                    use_container_width=True,
                )
                # Image quality check (warning only, not rejection)
                from PIL import Image as _PIL_Image
                _pil_check = _PIL_Image.open(verification_file).convert("RGB")
                verification_file.seek(0)  # reset buffer after reading
                _ok, _warns = _check_image_quality(_pil_check)
                for w in _warns:
                    st.warning(f"⚠️ Image quality note: {w}")
        else:
            verification_file = st.file_uploader(
                "Upload Verification Video",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                key="verification_video_uploader",
                help="Upload a video (.mp4, .avi, .mov, .mkv, .webm) for frame-based analysis",
            )
            if verification_file is not None:
                st.video(verification_file)
        glass_card_close()

    section_divider()

    verify = st.button("🔍  Verify Identity", use_container_width=True, type="primary")

    # -------------------- Run Pipeline --------------------

    if verify:

        # ---------------- Error Handling & Validation ----------------
        if reference_file is None:
            st.error("Please provide a reference image (Upload or take a live photo) before running verification.")
            return

        if verification_file is None:
            input_label = "video" if is_video_input else "image"
            st.error(f"Please provide a verification {input_label} before running verification.")
            return

        # Unique-per-run prefix so concurrent users/sessions never clobber
        # each other's temp files.
        session_id = st.session_state.setdefault("_ns_session_id", uuid.uuid4().hex[:12])
        run_id = uuid.uuid4().hex[:8]
        temp_prefix = f"{session_id}_{run_id}"

        reference_path = TEMP_DIR / f"{temp_prefix}_reference.jpg"
        try:
            with open(reference_path, "wb") as f:
                f.write(reference_file.getbuffer())
        except Exception as e:
            st.error(f"Failed to process reference image: {str(e)}")
            return

        verification_path = None
        temp_video_path = None
        if not is_video_input:
            verification_path = TEMP_DIR / f"{temp_prefix}_verification.jpg"
            try:
                with open(verification_path, "wb") as f:
                    f.write(verification_file.getbuffer())
                target_verification_input = str(verification_path)
            except Exception as e:
                st.error(f"Failed to process verification image: {str(e)}")
                reference_path.unlink(missing_ok=True)
                return
        else:
            # Temporary storage for video processing - preserve original extension
            file_name = getattr(verification_file, "name", "upload.mp4")
            file_ext = Path(file_name).suffix.lower()
            allowed_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
            if not file_ext or file_ext not in allowed_exts:
                file_ext = ".mp4"
            temp_video_path = TEMP_DIR / f"{temp_prefix}_verification_video{file_ext}"

            try:
                with open(temp_video_path, "wb") as f:
                    f.write(verification_file.getbuffer())
                target_verification_input = str(temp_video_path)
            except Exception as e:
                st.error(f"Failed to save temporary video for processing: {str(e)}")
                reference_path.unlink(missing_ok=True)
                return

        progress_bar = st.progress(0, text="Initializing verification pipeline...")
        skeleton_slot = st.empty()

        with skeleton_slot.container():
            skeleton(height_px=80, count=3)

        if is_video_input:
            steps = [
                (15, "Opening video & extracting representative frames (OpenCV)..."),
                (35, "Running frame-level deepfake inference (EfficientNet-B0)..."),
                (55, "Selecting representative frame & running face matching (FaceNet)..."),
                (70, "Computing multi-factor trust score..."),
                (85, "Generating Groth16 zero-knowledge proof..."),
                (95, "Writing to blockchain ledger & computing Grad-CAM..."),
            ]
        else:
            steps = [
                (20, "Running deepfake detection (EfficientNet-B0, multi-crop)..."),
                (45, "Matching facial features (FaceNet)..."),
                (65, "Computing trust score..."),
                (80, "Generating zero-knowledge proof..."),
                (95, "Writing to blockchain ledger & computing Grad-CAM..."),
            ]

        for pct, label in steps[:-1]:
            time.sleep(0.12)
            progress_bar.progress(pct, text=label)

        final_pct, final_label = steps[-1]
        progress_bar.progress(final_pct, text=final_label)

        # Execute Pipeline with Clean Error Handling
        try:
            with st.spinner(final_label):
                result = run_pipeline(
                    str(reference_path),
                    target_verification_input,
                    is_video=is_video_input,
                )
        except Exception as e:
            progress_bar.empty()
            skeleton_slot.empty()
            st.error(f"Verification Error: {str(e)}")
            return
        finally:
            for p in (temp_video_path, reference_path, verification_path):
                if p and p.exists():
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        progress_bar.progress(100, text="Verification complete.")
        time.sleep(0.15)
        progress_bar.empty()
        skeleton_slot.empty()

        st.success("✅ Verification Completed Successfully")
        section_divider()

        st.header("📊 Verification Results")

        # ---------------- Extracted Frames (For Video Verification) ----------------
        if result.get("is_video") and "extracted_frames" in result and result["extracted_frames"]:
            glass_card_open()
            st.markdown("#### 🎞️ Extracted Frames")
            st.caption(
                "Representative frames extracted across the video and evaluated "
                "using EfficientNet-B0 (multi-crop inference):"
            )

            frames = result["extracted_frames"]
            cols = st.columns(len(frames))
            rep_frame = result.get("representative_frame")
            frame_eval_map = {
                f["frame_number"]: f
                for f in result.get("deepfake", {}).get("frames", [])
            }

            for i, f_info in enumerate(frames):
                with cols[i]:
                    f_num = f_info["frame_number"]
                    f_path = f_info["path"]
                    is_rep = False
                    if rep_frame:
                        is_rep = Path(f_path).resolve() == Path(rep_frame).resolve()

                    st.image(f_path, use_container_width=True)
                    if is_rep:
                        st.markdown(f"<div style='color:#00E676;font-weight:700;font-size:0.92rem;margin-top:6px;'>Frame {i+1} ⭐ <i>(Face Rep)</i></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='color:#F8FAFC;font-weight:600;font-size:0.92rem;margin-top:6px;'>Frame {i+1}</div>", unsafe_allow_html=True)

                    if f_num in frame_eval_map:
                        feval = frame_eval_map[f_num]
                        p_badge = "success" if feval["prediction"] == "REAL" else "danger"
                        render_badge(
                            f"{feval['prediction']} ({feval['confidence']:.1f}%)",
                            p_badge,
                        )
                        face_icon = "👤" if feval.get("face_detected") else "❓"
                        st.markdown(f"<div style='color:#A7B0C0;font-size:0.82rem;margin-top:4px;'>{face_icon} Face: {'Yes' if feval.get('face_detected') else 'No'}</div>", unsafe_allow_html=True)

            # Video-level stats
            ddf = result["deepfake"]
            st.markdown("---")
            meta_cols = st.columns(4)
            meta_cols[0].metric("Frames Analysed", ddf.get("total_frames_analyzed", "—"))
            meta_cols[1].metric("Valid Face Frames", ddf.get("valid_face_frames", "—"))
            meta_cols[2].metric(
                "Avg FAKE Prob",
                f"{ddf.get('avg_fake_probability', 0):.1f}%",
            )
            meta_cols[3].metric(
                "Max FAKE Prob",
                f"{ddf.get('max_fake_probability', 0):.1f}%",
            )

            glass_card_close()
            st.write("")

        # ---------------- Deepfake + Face Verification ----------------
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            glass_card_open()
            st.markdown("#### 🧠 Deepfake Detection")
            prediction = result["deepfake"]["prediction"]
            badge_status = "success" if str(prediction).upper() == "REAL" else "danger"
            render_badge(prediction, badge_status)
            st.write("")
            labeled_progress("Confidence", result["deepfake"]["confidence"])

            # Confidence state display
            confidence_state = result["deepfake"].get("confidence_state", "")
            fake_prob = result["deepfake"].get("fake_probability", 0.0)
            if confidence_state:
                _render_confidence_state(confidence_state, fake_prob)

            # Detailed probabilities
            real_prob_pct = result["deepfake"].get("real_probability", 0.0) * 100.0
            fake_prob_pct = fake_prob * 100.0
            st.write(
                f"REAL: **{real_prob_pct:.2f}%** | "
                f"FAKE: **{fake_prob_pct:.2f}%** | "
                f"Threshold: **{result.get('decision_threshold', 0.5):.2f}**"
            )

            if result.get("is_video"):
                st.markdown(
                    "<hr style='margin:12px 0;border-color:#E2E8F0;'/>",
                    unsafe_allow_html=True,
                )
                st.markdown("**Frame-based Video Analysis**")
                ddf = result["deepfake"]
                avg_fake = ddf.get("avg_fake_probability", 0.0)
                max_fake = ddf.get("max_fake_probability", 0.0)
                st.write(f"Average FAKE Probability: **{avg_fake:.2f}%**")
                st.write(f"Maximum FAKE Probability: **{max_fake:.2f}%**")
                labeled_progress("Avg Fake Probability", avg_fake)
            glass_card_close()

        with r1c2:
            glass_card_open()
            st.markdown("#### 👤 Face Verification")
            verified = result["verification"]["verified"]
            render_badge(
                "VERIFIED" if verified else "NOT VERIFIED",
                "success" if verified else "danger",
            )
            st.write("")
            labeled_progress("Similarity Score", result["verification"]["similarity_score"])
            if result["verification"].get("error"):
                st.warning(result["verification"]["error"])
            if result.get("is_video") and result.get("representative_frame"):
                st.caption(
                    f"Representative frame used: `{Path(result['representative_frame']).name}`"
                )
            glass_card_close()

        st.write("")

        # ---------------- Trust Score + ZKP ----------------
        r2c1, r2c2 = st.columns(2)

        with r2c1:
            glass_card_open()
            st.markdown("#### 🛡️ Trust Score")
            circular_gauge(result["trust_score"], label="Trust Score")
            st.write("")
            if result["status"] == "TRUSTED":
                render_badge("✅ TRUSTED", "success")
            elif result["status"] == "SUSPICIOUS":
                render_badge("⚠️ SUSPICIOUS", "warning")
            else:
                render_badge("❌ REJECTED", "danger")
            glass_card_close()

        with r2c2:
            glass_card_open()
            st.markdown("#### 🔐 Zero-Knowledge Proof")
            zkp = result["zkp"]
            render_badge(
                "Proof Generated  ✓" if zkp["generated"] else "Proof Generation Failed  ✗",
                "success" if zkp["generated"] else "danger",
            )
            st.write("")
            render_badge(
                "Proof Verified  ✓" if zkp["verified"] else "Proof Verification Failed  ✗",
                "success" if zkp["verified"] else "danger",
            )
            st.write("")
            render_badge(
                "Policy Passed  ✓" if zkp["policy_passed"] else "Policy Failed  ⚠️",
                "success" if zkp["policy_passed"] else "warning",
            )
            glass_card_close()

        st.write("")

        # ---------------- Blockchain ----------------
        glass_card_open()
        st.markdown("#### ⛓️ Blockchain Proof")
        code_card("Proof Hash", result["blockchain"]["proof_hash"] or "None")
        st.write("")
        if result["blockchain"]["proof_hash"]:
            render_badge("Stored Successfully on Blockchain", "success")
        else:
            render_badge("Ledger Storage Skipped (Policy Failed)", "warning")
        glass_card_close()

        st.write("")

        # ---------------- Grad-CAM ----------------
        glass_card_open()
        st.markdown("#### 🔥 Explainable AI — Grad-CAM")
        if result.get("is_video"):
            st.caption("Grad-CAM visualization generated for the representative frame.")
        gradcam_image = result.get("gradcam", {}).get("image")
        if gradcam_image and Path(gradcam_image).exists():
            st.image(gradcam_image, caption="Grad-CAM Heatmap", use_container_width=True)
        else:
            st.warning("Grad-CAM visualization could not be generated for this image.")
        glass_card_close()

        section_divider()

        # ---------------- Debug / Diagnostics ----------------
        show_debug = st.checkbox("🔬 Show detection diagnostics", value=False)
        if show_debug:
            glass_card_open()
            st.markdown("#### 🔬 Detection Diagnostics")
            ddf = result["deepfake"]
            diag_data = {
                "Input image size": str(ddf.get("input_size", "N/A")),
                "Processed size": str(ddf.get("processed_size", (224, 224))),
                "REAL probability": f"{ddf.get('real_probability', 0.0) * 100:.4f}%",
                "FAKE probability": f"{ddf.get('fake_probability', 0.0) * 100:.4f}%",
                "Decision threshold": str(result.get("decision_threshold", 0.50)),
                "Confidence state": ddf.get("confidence_state", "N/A"),
                "Method": ddf.get("method", "N/A"),
                "Crops used": str(ddf.get("crops_used", "N/A")),
            }

            if "crop_labels" in ddf and "crop_fake_probs" in ddf:
                for lbl, prob in zip(ddf["crop_labels"], ddf["crop_fake_probs"]):
                    diag_data[f"  {lbl}"] = f"FAKE {prob * 100:.2f}%"

            if ddf.get("is_video") or result.get("is_video"):
                diag_data["Total frames analysed"] = str(ddf.get("total_frames_analyzed", "N/A"))
                diag_data["Valid face frames"] = str(ddf.get("valid_face_frames", "N/A"))
                diag_data["Avg FAKE probability"] = f"{ddf.get('avg_fake_probability', 0):.2f}%"
                diag_data["Max FAKE probability"] = f"{ddf.get('max_fake_probability', 0):.2f}%"

                # Video decoder diagnostics
                dec_debug = ddf.get("decoder_debug", {})
                if dec_debug:
                    diag_data["Video Original Filename"] = str(dec_debug.get("original_filename", "N/A"))
                    diag_data["Video File Extension"] = str(dec_debug.get("file_extension", "N/A"))
                    diag_data["Video File Size"] = f"{dec_debug.get('file_size_bytes', 0):,} bytes"
                    diag_data["OpenCV Direct Open"] = "Yes" if dec_debug.get("opencv_direct_open") else "No"
                    diag_data["Fallback Conversion Used"] = "Yes" if dec_debug.get("fallback_conversion_used") else "No"
                    if dec_debug.get("converted_file"):
                        diag_data["Converted File"] = str(dec_debug.get("converted_file"))
                    diag_data["Video Total Frames"] = str(dec_debug.get("num_frames", "N/A"))
                    diag_data["Video FPS"] = f"{dec_debug.get('fps', 0):.1f}"
                    diag_data["Video Duration"] = f"{dec_debug.get('duration_sec', 0):.2f}s"

            for k, v in diag_data.items():
                st.write(f"**{k}:** {v}")
            glass_card_close()

        section_divider()

        # ---------------- Report Download ----------------
        try:
            pdf_path = generate_pdf(result)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Download Verification Report",
                    data=pdf_file,
                    file_name="verification_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        except Exception as e:
            st.warning(f"Could not generate PDF report: {str(e)}")

    # -------------------- Real Camera Test Mode --------------------
    section_divider()
    with st.expander("📷 Real Camera Test Mode (for evaluating false-positive rate)", expanded=False):
        _render_real_camera_test()