import streamlit as st
from components.ui import fade, stat_card, feature_card, section_divider


def render():

    # ---------------- Hero ----------------
    st.markdown(
        """
        <div class="ns-hero ns-fade">
            <div class="ns-pill"><span class="ns-pill-dot"></span> Live · Enterprise Identity Security Platform</div>
            <div class="ns-hero-title">NeuroShield-ID</div>
            <div class="ns-hero-sub">
                AI-powered secure identity verification combining deepfake detection, biometric matching,
                zero-knowledge proofs, and blockchain audit trails into a single trusted verification pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # ---------------- Quick Stats ----------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("🧩", "Active Modules", "8", delay=1)
    with c2:
        stat_card("🧠", "AI Models", "3", delay=2)
    with c3:
        stat_card("🔐", "Security Layers", "ZKP + Chain", delay=3)
    with c4:
        stat_card("⚡", "Avg. Inference", "< 2s", delay=4)

    section_divider()

    # ---------------- Feature Grid ----------------
    st.subheader("Platform Capabilities")

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        feature_card("🧠", "Deepfake Detection", "EfficientNet-B0 powered image authenticity scoring.", 1)
    with r1c2:
        feature_card("👤", "Face Verification", "DeepFace / FaceNet biometric similarity matching.", 2)
    with r1c3:
        feature_card("🛡️", "AI Trust Score", "Unified confidence score fusing multiple signals.", 3)

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        feature_card("🔐", "Zero-Knowledge Proof", "Privacy-preserving cryptographic verification.", 4)
    with r2c2:
        feature_card("⛓️", "Blockchain Ledger", "Immutable, persistent proof-of-verification storage.", 5)
    with r2c3:
        feature_card("🔥", "Explainable AI", "Grad-CAM visual reasoning for every decision.", 6)

    section_divider()

    # ---------------- Architecture Overview ----------------
    st.subheader("Verification Pipeline")

    steps = [
        ("📥", "Image Intake"),
        ("🧠", "Deepfake Scan"),
        ("👤", "Face Match"),
        ("🛡️", "Trust Score"),
        ("🔐", "ZK Proof"),
        ("⛓️", "Blockchain"),
        ("📄", "Report"),
    ]

    cols = st.columns(len(steps))
    for i, (icon, label) in enumerate(steps):
        with cols[i]:
            st.markdown(
                fade(
                    f"""
                    <div class="ns-card" style="text-align:center;padding:16px 8px;">
                        <div style="font-size:1.4rem;">{icon}</div>
                        <div style="font-size:.78rem;margin-top:6px;color:var(--text-mid);">{label}</div>
                    </div>
                    """,
                    (i % 6) + 1,
                ),
                unsafe_allow_html=True,
            )

    st.write("")
    st.info("👈 Use the sidebar to open **Verification** and run a live identity check.")
