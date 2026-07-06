import streamlit as st
import pandas as pd

from src.database.database import load_history

st.set_page_config(

    page_title="Verification History",

    page_icon="📜",

    layout="wide"

)

st.title("📜 Verification History")

rows = load_history()

if len(rows)==0:

    st.info("No verification history found.")

else:

    df=pd.DataFrame(

        rows,

        columns=[

            "ID",

            "Timestamp",

            "Prediction",

            "Confidence",

            "Similarity",

            "Trust Score",

            "Status",

            "Proof Hash"

        ]

    )

    st.dataframe(

        df,

        width="stretch",

        hide_index=True

    )

    csv=df.to_csv(index=False)

    st.download_button(

        "⬇ Download CSV",

        csv,

        "verification_history.csv",

        "text/csv",

        width="stretch"

    )