import streamlit as st
from pathlib import Path
import time

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


def render():

    page_header("🔍", "Identity Verification", "Upload a reference and verification image to run the full security pipeline.")

    # -------------------- Upload Cards --------------------

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📎 Reference Image")
        reference_file = st.file_uploader(
            "Drag & drop or browse",
            type=["jpg", "jpeg", "png"],
            key="reference",
            label_visibility="collapsed",
        )
        if reference_file:
            st.image(reference_file, use_container_width=True)

    with col2:
        st.markdown("#### 📎 Verification Image")
        verification_file = st.file_uploader(
            "Drag & drop or browse",
            type=["jpg", "jpeg", "png"],
            key="verification",
            label_visibility="collapsed",
        )
        if verification_file:
            st.image(verification_file, use_container_width=True)

    section_divider()

    verify = st.button("🔍  Verify Identity", use_container_width=True)

    # -------------------- Run Pipeline --------------------

    if verify:

        if reference_file is None or verification_file is None:
            st.error("Please upload both images before running verification.")
            return

        reference_path = TEMP_DIR / "reference.jpg"
        verification_path = TEMP_DIR / "verification.jpg"

        with open(reference_path, "wb") as f:
            f.write(reference_file.getbuffer())

        with open(verification_path, "wb") as f:
            f.write(verification_file.getbuffer())

        progress_bar = st.progress(0, text="Initializing pipeline...")
        skeleton_slot = st.empty()

        with skeleton_slot.container():
            skeleton(height_px=80, count=3)

        for pct, label in [
            (20, "Running deepfake detection..."),
            (45, "Matching facial features..."),
            (65, "Computing trust score..."),
            (80, "Generating zero-knowledge proof..."),
            (95, "Writing to blockchain ledger..."),
        ]:
            time.sleep(0.15)
            progress_bar.progress(pct, text=label)

        result = run_pipeline(str(reference_path), str(verification_path))

        progress_bar.progress(100, text="Verification complete.")
        time.sleep(0.2)
        progress_bar.empty()
        skeleton_slot.empty()

        st.success("✅ Verification Completed Successfully")
        section_divider()

        st.header("📊 Verification Results")

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
            glass_card_close()

        with r1c2:
            glass_card_open()
            st.markdown("#### 👤 Face Verification")
            verified = result["verification"]["verified"]
            render_badge("VERIFIED" if verified else "NOT VERIFIED", "success" if verified else "danger")
            st.write("")
            labeled_progress("Similarity Score", result["verification"]["similarity_score"])
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
            else:
                render_badge("⚠️ SUSPICIOUS", "warning")
            glass_card_close()

        with r2c2:
            glass_card_open()
            st.markdown("#### 🔐 Zero-Knowledge Proof")
            zkp = result["zkp"]
            render_badge("Proof Generated" if zkp["generated"] else "Generation Failed",
                         "success" if zkp["generated"] else "danger")
            st.write("")
            render_badge("Proof Verified" if zkp["verified"] else "Verification Failed",
                         "success" if zkp["verified"] else "danger")
            st.write("")
            render_badge("Policy Passed" if zkp["policy_passed"] else "Policy Failed",
                         "success" if zkp["policy_passed"] else "warning")
            glass_card_close()

        st.write("")

        # ---------------- Blockchain ----------------
        glass_card_open()
        st.markdown("#### ⛓️ Blockchain Proof")
        code_card("Proof Hash", result["blockchain"]["proof_hash"])
        st.write("")
        render_badge("Stored Successfully on Blockchain", "success")
        glass_card_close()

        st.write("")

        # ---------------- Grad-CAM ----------------
        glass_card_open()
        st.markdown("#### 🔥 Explainable AI — Grad-CAM")
        st.image(result["gradcam"]["image"], caption="Grad-CAM Heatmap", use_container_width=True)
        glass_card_close()

        section_divider()

        # ---------------- Report Download ----------------
        pdf_path = generate_pdf(result)

        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download Verification Report",
                data=pdf_file,
                file_name="verification_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
