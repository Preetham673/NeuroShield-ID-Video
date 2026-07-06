import streamlit as st

from components.ui import page_header, section_divider, feature_card


def render():

    page_header("ℹ️", "About NeuroShield-ID", "AI-powered identity verification platform, built for trust and transparency.")

    st.markdown(
        """
        <div class="ns-card ns-fade">
            <p style="margin:0;">
            NeuroShield-ID fuses deepfake detection, biometric face verification, zero-knowledge proofs,
            and blockchain audit trails into a single verification pipeline — combining AI accuracy with
            cryptographic trust guarantees.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_divider()

    st.subheader("Core Features")

    r1 = st.columns(3)
    with r1[0]:
        feature_card("🧠", "Deepfake Detection", "EfficientNet-B0 classification of image authenticity.", 1)
    with r1[1]:
        feature_card("👤", "Face Verification", "DeepFace / FaceNet biometric similarity scoring.", 2)
    with r1[2]:
        feature_card("🛡️", "Trust Score", "Composite score fusing multiple verification signals.", 3)

    r2 = st.columns(3)
    with r2[0]:
        feature_card("🔥", "Explainable AI", "Grad-CAM visual reasoning for model decisions.", 4)
    with r2[1]:
        feature_card("🔐", "Zero-Knowledge Proof", "Privacy-preserving cryptographic verification.", 5)
    with r2[2]:
        feature_card("⛓️", "Blockchain Audit Trail", "Immutable, persistent proof storage.", 6)

    section_divider()

    st.subheader("Technology Stack")

    stack_cols = st.columns(4)
    stack = ["Python", "PyTorch", "TensorFlow", "DeepFace", "Streamlit", "Circom", "SnarkJS", "ReportLab"]
    for i, tech in enumerate(stack):
        with stack_cols[i % 4]:
            st.markdown(
                f'<div class="ns-card ns-fade" style="text-align:center;padding:14px;"><b>{tech}</b></div>',
                unsafe_allow_html=True,
            )

    section_divider()

    st.subheader("Verification Pipeline")

    steps = [
        "Reference Image Intake",
        "Deepfake Detection",
        "Face Verification",
        "Trust Score Calculation",
        "Zero-Knowledge Proof",
        "Blockchain Storage",
        "Verification Report",
    ]

    for i, step in enumerate(steps):
        st.markdown(
            f"""
            <div class="ns-timeline-item{' latest' if i == len(steps)-1 else ''}">
                <b>{i+1}. {step}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_divider()
    st.caption("Developed as a secure AI-powered identity verification system.")
