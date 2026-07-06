import streamlit as st
from streamlit_option_menu import option_menu

from components.ui import inject_css, footer

from modules import (
    home,
    verification,
    analytics,
    model_evaluation,
    blockchain,
    history,
    reports,
    about,
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="NeuroShield-ID | Enterprise Identity Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# --------------------------------------------------
# Sidebar — Enterprise Navigation
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;padding:6px 4px 18px 4px;">
            <div style="font-size:1.7rem;">🛡️</div>
            <div>
                <div style="font-weight:800;font-size:1.05rem;color:var(--text-hi);line-height:1.1;">NeuroShield-ID</div>
                <div style="font-size:.72rem;color:var(--text-lo);letter-spacing:.04em;">ENTERPRISE SECURITY</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = option_menu(
        menu_title=None,
        options=[
            "Home",
            "Verification",
            "Analytics",
            "Model Evaluation",
            "Blockchain Explorer",
            "Verification History",
            "Reports",
            "About",
        ],
        icons=[
            "house",
            "search",
            "bar-chart",
            "graph-up",
            "link-45deg",
            "clock-history",
            "file-earmark-text",
            "info-circle",
        ],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"font-size": "16px"},
            "nav-link": {
                "font-size": "14.5px",
                "font-weight": "500",
                "text-align": "left",
                "margin": "3px 0",
                "border-radius": "10px",
                "padding": "10px 14px",
                "color": "#9aa7bd",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, rgba(37,99,235,0.35), rgba(124,58,237,0.30))",
                "color": "#eef2f9",
                "font-weight": "600",
                "box-shadow": "0 0 0 1px rgba(99,179,255,0.25)",
            },
        },
    )

    st.markdown('<hr class="ns-divider"/>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="padding:0 4px;">
            <span class="ns-badge ns-badge-success">● System Operational</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# Page Router
# --------------------------------------------------

PAGES = {
    "Home": home.render,
    "Verification": verification.render,
    "Analytics": analytics.render,
    "Model Evaluation": model_evaluation.render,
    "Blockchain Explorer": blockchain.render,
    "Verification History": history.render,
    "Reports": reports.render,
    "About": about.render,
}

PAGES[selected]()

footer()