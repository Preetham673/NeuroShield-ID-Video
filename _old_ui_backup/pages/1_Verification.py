import streamlit as st
from pathlib import Path
from src.reports.pdf_report import generate_pdf

from src.pipeline.pipeline import run_pipeline

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="NeuroShield-ID",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# Temporary Folder
# --------------------------------------------------

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🛡️ NeuroShield-ID")
st.caption("AI-Powered Secure Identity Verification System")

st.divider()

# --------------------------------------------------
# Image Upload Section
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Reference Image")

    reference_file = st.file_uploader(
        "Upload Reference Image",
        type=["jpg", "jpeg", "png"],
        key="reference"
    )

    if reference_file:
        st.image(reference_file, use_container_width=True)

with col2:

    st.subheader("Verification Image")

    verification_file = st.file_uploader(
        "Upload Verification Image",
        type=["jpg", "jpeg", "png"],
        key="verification"
    )

    if verification_file:
        st.image(verification_file, use_container_width=True)

st.divider()

# --------------------------------------------------
# Verify Button
# --------------------------------------------------

verify = st.button(
    "🔍 Verify Identity",
    use_container_width=True
)

# --------------------------------------------------
# Run Pipeline
# --------------------------------------------------

if verify:

    if reference_file is None or verification_file is None:

        st.error("Please upload both images.")

    else:

        # Save uploaded images

        reference_path = TEMP_DIR / "reference.jpg"
        verification_path = TEMP_DIR / "verification.jpg"

        with open(reference_path, "wb") as f:
            f.write(reference_file.getbuffer())

        with open(verification_path, "wb") as f:
            f.write(verification_file.getbuffer())

        # Run Backend

        with st.spinner("Running NeuroShield-ID Pipeline..."):

            result = run_pipeline(
                str(reference_path),
                str(verification_path)
            )

        st.success("Verification Completed Successfully!")

        st.divider()

        # ==================================================
        # RESULTS
        # ==================================================

        st.header("📊 Verification Results")

        # -----------------------------
        # Deepfake Detection
        # -----------------------------

        st.subheader("🧠 Deepfake Detection")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Prediction",
                result["deepfake"]["prediction"]
            )

        with col2:
            st.metric(
                "Confidence",
                f'{result["deepfake"]["confidence"]:.2f}%'
            )

        st.divider()

        # -----------------------------
        # Face Verification
        # -----------------------------

        st.subheader("👤 Face Verification")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Verified",
                str(result["verification"]["verified"])
            )

        with col2:
            st.metric(
                "Similarity",
                f'{result["verification"]["similarity_score"]:.2f}%'
            )

        st.divider()

        # -----------------------------
        # Trust Score
        # -----------------------------

        st.subheader("🛡️ Trust Score")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Trust Score",
                f'{result["trust_score"]:.2f}'
            )

        with col2:

            if result["status"] == "TRUSTED":
                st.success("✅ TRUSTED")
            else:
                st.warning("⚠️ SUSPICIOUS")

        st.divider()

        # -----------------------------
        # Zero Knowledge Proof
        # -----------------------------

        st.subheader("🔐 Zero-Knowledge Proof")

        if result["zkp"]["generated"]:
            st.success("✅ Proof Generated")
        else:
            st.error("❌ Proof Generation Failed")

        if result["zkp"]["verified"]:
            st.success("✅ Proof Verified")
        else:
            st.error("❌ Proof Verification Failed")

        if result["zkp"]["policy_passed"]:
            st.success("🟢 Security Policy Passed")
        else:
            st.warning("🟡 Security Policy Failed")

        st.divider()

        # -----------------------------
        # Blockchain
        # -----------------------------

        st.subheader("⛓️ Blockchain")

        st.text_area(
            "Proof Hash",
            result["blockchain"]["proof_hash"],
            height=80
        )

        st.success("Stored Successfully on Blockchain")
        st.divider()

        st.subheader("🔥 Explainable AI (Grad-CAM)")

        st.image(
        result["gradcam"]["image"],
        caption="Grad-CAM Heatmap",
        width="stretch"
        )
        st.divider()

        pdf_path = generate_pdf(result)

        with open(pdf_path, "rb") as pdf_file:

         st.download_button(
        label="📄 Download Verification Report",
        data=pdf_file,
        file_name="verification_report.pdf",
        mime="application/pdf",
        width="stretch"
    )