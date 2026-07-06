import streamlit as st
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Reports",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Verification Reports")

st.caption("Download and manage generated verification reports.")

REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)

pdf_files = sorted(
    REPORT_DIR.glob("*.pdf"),
    reverse=True
)

if len(pdf_files) == 0:

    st.info("No reports generated yet.")

else:

    st.success(f"{len(pdf_files)} report(s) found.")

    st.divider()

    for pdf in pdf_files:

        modified = datetime.fromtimestamp(
            pdf.stat().st_mtime
        )

        with st.expander(f"📄 {pdf.name}"):

            st.write(
                f"**Created:** {modified.strftime('%d-%m-%Y %H:%M:%S')}"
            )

            st.write(
                f"**Size:** {round(pdf.stat().st_size/1024,2)} KB"
            )

            with open(pdf, "rb") as f:

                st.download_button(
                    label="⬇ Download Report",
                    data=f,
                    file_name=pdf.name,
                    mime="application/pdf",
                    width="stretch"
                )