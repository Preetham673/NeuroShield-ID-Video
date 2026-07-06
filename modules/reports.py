import streamlit as st
from pathlib import Path
from datetime import datetime

from components.ui import page_header, section_divider, stat_card, glass_card_open, glass_card_close

REPORT_DIR = Path("reports")


def render():

    page_header("📄", "Verification Reports", "Download and manage generated verification reports.")

    REPORT_DIR.mkdir(exist_ok=True)

    pdf_files = sorted(REPORT_DIR.glob("*.pdf"), reverse=True)

    if len(pdf_files) == 0:
        st.info("No reports generated yet.")
        return

    total_size_kb = sum(p.stat().st_size for p in pdf_files) / 1024

    c1, c2 = st.columns(2)
    with c1:
        stat_card("📄", "Total Reports", str(len(pdf_files)), 1)
    with c2:
        stat_card("💾", "Total Size", f"{total_size_kb:.1f} KB", 2)

    section_divider()

    search_term = st.text_input("🔎 Search reports by filename", "")

    filtered = [p for p in pdf_files if search_term.lower() in p.name.lower()] if search_term else pdf_files

    st.caption(f"Showing {len(filtered)} of {len(pdf_files)} report(s)")
    st.write("")

    cols = st.columns(3)

    for i, pdf in enumerate(filtered):
        modified = datetime.fromtimestamp(pdf.stat().st_mtime)
        size_kb = round(pdf.stat().st_size / 1024, 2)

        with cols[i % 3]:
            glass_card_open()
            st.markdown(f"**📄 {pdf.name}**")
            st.caption(f"Created: {modified.strftime('%d-%m-%Y %H:%M:%S')}")
            st.caption(f"Size: {size_kb} KB")

            with open(pdf, "rb") as f:
                st.download_button(
                    label="⬇ Download",
                    data=f,
                    file_name=pdf.name,
                    mime="application/pdf",
                    key=f"download_{pdf.name}",
                    use_container_width=True,
                )

            if st.button("🗑 Delete", key=f"delete_{pdf.name}", use_container_width=True):
                pdf.unlink(missing_ok=True)
                st.rerun()

            glass_card_close()
