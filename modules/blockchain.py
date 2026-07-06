import streamlit as st
import pandas as pd

from src.blockchain.persistent_blockchain import PersistentBlockchain

from components.ui import page_header, section_divider, stat_card, code_card, render_badge


def render():

    page_header("⛓️", "Blockchain Explorer", "Persistent, tamper-evident ledger of verification proofs.")

    blockchain = PersistentBlockchain()
    chain = blockchain.chain

    c1, c2, c3 = st.columns(3)
    with c1:
        stat_card("🧱", "Total Blocks", str(len(chain)), 1)
    with c2:
        stat_card("🌱", "Genesis Hash", f"{chain[0]['hash'][:14]}...", 2)
    with c3:
        stat_card("🔗", "Latest Hash", f"{chain[-1]['hash'][:14]}...", 3)

    section_divider()

    search_term = st.text_input("🔎 Search by hash", placeholder="Paste a full or partial block hash...")

    st.write("")

    display_chain = list(reversed(chain))
    if search_term:
        display_chain = [b for b in display_chain if search_term.lower() in b["hash"].lower()]
        if not display_chain:
            st.warning("No blocks match that hash.")

    for i, block in enumerate(display_chain):
        is_latest = block["index"] == chain[-1]["index"]

        with st.expander(f"🧱 Block #{block['index']}" + ("  •  🟢 LATEST" if is_latest else ""), expanded=is_latest and i == 0):
            st.write(f"**Timestamp:** {block['timestamp']}")
            st.write("**Data**")
            st.code(block["data"])
            code_card("Previous Hash", block["previous_hash"])
            st.write("")
            code_card("Current Hash", block["hash"])

    section_divider()

    # ---------------- Summary ----------------
    st.markdown("#### Blockchain Summary")

    summary = pd.DataFrame({
        "Property": ["Total Blocks", "Genesis Block", "Latest Block"],
        "Value": [
            len(chain),
            chain[0]["hash"][:20] + "...",
            chain[-1]["hash"][:20] + "...",
        ],
    })

    st.dataframe(summary, hide_index=True, use_container_width=True)
