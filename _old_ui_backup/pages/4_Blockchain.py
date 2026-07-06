import streamlit as st
import pandas as pd

from src.blockchain.persistent_blockchain import PersistentBlockchain

st.set_page_config(
    page_title="Blockchain Explorer",
    page_icon="⛓️",
    layout="wide"
)

st.title("⛓️ Blockchain Explorer")

st.caption("Persistent Blockchain Ledger")

blockchain = PersistentBlockchain()

chain = blockchain.chain

st.success(f"Total Blocks : {len(chain)}")

st.divider()

for block in reversed(chain):

    with st.expander(f"🧱 Block #{block['index']}"):

        st.write(f"**Timestamp:** {block['timestamp']}")

        st.write(f"**Data:**")

        st.code(block["data"])

        st.write("**Previous Hash**")

        st.code(block["previous_hash"])

        st.write("**Current Hash**")

        st.code(block["hash"])

st.divider()

summary = pd.DataFrame({

    "Property":[

        "Total Blocks",

        "Genesis Block",

        "Latest Block"

    ],

    "Value":[

        len(chain),

        chain[0]["hash"][:20]+"...",

        chain[-1]["hash"][:20]+"..."

    ]

})

st.subheader("Blockchain Summary")

st.dataframe(

    summary,

    hide_index=True,

    width="stretch"

)