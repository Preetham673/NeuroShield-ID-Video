import streamlit as st
import pandas as pd

from src.database.database import load_history

from components.ui import page_header, section_divider, stat_card


def render():

    page_header("📜", "Verification History", "Full audit trail of past identity verification runs.")

    rows = load_history()

    if len(rows) == 0:
        st.info("No verification history found.")
        return

    df = pd.DataFrame(
        rows,
        columns=[
            "ID",
            "Timestamp",
            "Prediction",
            "Confidence",
            "Similarity",
            "Trust Score",
            "Status",
            "Proof Hash",
        ],
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        stat_card("📋", "Total Records", str(len(df)), 1)
    with c2:
        trusted_count = int((df["Status"] == "TRUSTED").sum())
        stat_card("✅", "Trusted", str(trusted_count), 2)
    with c3:
        avg_trust = df["Trust Score"].astype(float).mean() if len(df) else 0
        stat_card("🛡️", "Avg Trust Score", f"{avg_trust:.2f}", 3)

    section_divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("🔎 Search by prediction, status, or hash", "")
    with col2:
        status_filter = st.selectbox("Filter by status", ["All"] + sorted(df["Status"].unique().tolist()))

    filtered = df.copy()
    if search_term:
        mask = filtered.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)
        filtered = filtered[mask]
    if status_filter != "All":
        filtered = filtered[filtered["Status"] == status_filter]

    st.caption(f"Showing {len(filtered)} of {len(df)} records")

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    csv = filtered.to_csv(index=False)

    st.download_button(
        "⬇ Download CSV",
        csv,
        "verification_history.csv",
        "text/csv",
        use_container_width=True,
    )
