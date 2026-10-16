"""
components/ui.py

Reusable, purely-presentational UI building blocks for NeuroShield-ID.
Enterprise Dark + Vibrant Green Cybersecurity Theme.
No backend/pipeline logic lives here — only rendering helpers.
"""

import streamlit as st
from pathlib import Path

CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "style.css"


def inject_css():
    """Load the global stylesheet once per session."""
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def fade(html: str, delay_index: int = 0) -> str:
    """Wrap a block of HTML with a staggered fade-in animation class."""
    cls = f"ns-fade ns-fade-{min(delay_index, 6)}" if delay_index else "ns-fade"
    return f'<div class="{cls}">{html}</div>'


def page_header(icon: str, title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="ns-fade" style="margin-bottom: 8px;">
            <div style="display:flex;align-items:center;gap:12px;">
                <span style="display:inline-flex;align-items:center;justify-content:center;width:42px;height:42px;background:rgba(0,230,118,0.12);border:1px solid rgba(0,230,118,0.35);border-radius:12px;font-size:1.4rem;color:#00E676;">{icon}</span>
                <div>
                    <h1 style="margin:0;font-size:1.85rem;font-weight:800;color:#F8FAFC;line-height:1.2;">{title}</h1>
                    <p style="margin:3px 0 0 0;color:#A7B0C0;font-size:0.95rem;">{subtitle}</p>
                </div>
            </div>
        </div>
        <hr class="ns-divider"/>
        """,
        unsafe_allow_html=True,
    )


def section_divider():
    st.markdown('<hr class="ns-divider"/>', unsafe_allow_html=True)


def status_badge(text: str, status: str = "info") -> str:
    """status: success | warning | danger | info"""
    return f'<span class="ns-badge ns-badge-{status}">{text}</span>'


def render_badge(text: str, status: str = "info"):
    st.markdown(status_badge(text, status), unsafe_allow_html=True)


_ICON_COLORS = ["ns-icon-emerald", "ns-icon-blue", "ns-icon-amber", "ns-icon-violet", "ns-icon-cyan", "ns-icon-pink"]


def stat_card(icon: str, label: str, value: str, delay: int = 0, color_index: int = None):
    color_cls = _ICON_COLORS[(color_index if color_index is not None else delay) % len(_ICON_COLORS)]
    html = f"""
    <div class="ns-card">
        <div class="ns-feature-icon {color_cls}">{icon}</div>
        <div class="ns-stat-value">{value}</div>
        <div class="ns-stat-label">{label}</div>
    </div>
    """
    st.markdown(fade(html, delay), unsafe_allow_html=True)


def feature_card(icon: str, title: str, desc: str, delay: int = 0, color_index: int = None):
    color_cls = _ICON_COLORS[(color_index if color_index is not None else delay) % len(_ICON_COLORS)]
    html = f"""
    <div class="ns-card">
        <div class="ns-feature-icon {color_cls}">{icon}</div>
        <h4 style="margin:4px 0;color:#F8FAFC;">{title}</h4>
        <p style="font-size:.88rem;margin:0;color:#A7B0C0;">{desc}</p>
    </div>
    """
    st.markdown(fade(html, delay), unsafe_allow_html=True)


def code_card(label: str, value: str):
    st.markdown(
        f"""
        <p style="font-size:.82rem;color:var(--text-mid);margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em;font-weight:600;">{label}</p>
        <div class="ns-code-card">{value}</div>
        """,
        unsafe_allow_html=True,
    )


def circular_gauge(value: float, label: str = "Trust Score", max_value: float = 100.0):
    """
    Renders a conic-gradient circular gauge.
    Positive portion: #00E676 (or warning/error based on thresholds)
    Remaining portion: #1F2937
    Thresholds:
    >= 75.0 -> #00E676 (TRUSTED)
    50.0 <= value < 75.0 -> #FFB020 (SUSPICIOUS)
    < 50.0 -> #FF4D5A (REJECTED)
    """
    pct = max(0.0, min(1.0, value / max_value))
    deg = pct * 360

    if pct >= 0.75:
        color = "#00E676"
        status_label = "TRUSTED"
        status_color = "#00E676"
    elif pct >= 0.50:
        color = "#FFB020"
        status_label = "SUSPICIOUS"
        status_color = "#FFB020"
    else:
        color = "#FF4D5A"
        status_label = "REJECTED"
        status_color = "#FF4D5A"

    html = f"""
    <div class="ns-gauge-wrap">
        <div class="ns-gauge" style="background: conic-gradient({color} {deg}deg, #1F2937 {deg}deg 360deg);">
            <div class="ns-gauge-inner">
                <div class="ns-gauge-value" style="color:#F8FAFC;">{value:.1f}</div>
                <div class="ns-gauge-label" style="color:{status_color};font-weight:700;">{status_label}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def labeled_progress(label: str, value: float, max_value: float = 100.0):
    pct = max(0.0, min(1.0, value / max_value))
    st.markdown(
        f'<p style="font-size:.85rem;margin-bottom:4px;color:var(--text-mid);">{label} '
        f'<span style="float:right;color:var(--text-hi);font-weight:700;">{value:.2f}%</span></p>',
        unsafe_allow_html=True,
    )
    st.progress(pct)


def skeleton(height_px: int = 90, count: int = 1):
    cols = st.columns(count)
    for c in cols:
        with c:
            st.markdown(
                f'<div class="ns-skeleton" style="height:{height_px}px;"></div>',
                unsafe_allow_html=True,
            )


def glass_card_open():
    st.markdown('<div class="ns-card ns-fade">', unsafe_allow_html=True)


def glass_card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def footer():
    st.markdown(
        """
        <div class="ns-footer">
            <span style="color:#00E676;font-weight:700;">🛡️ NeuroShield-ID</span> &nbsp;•&nbsp;
            Enterprise Cybersecurity Platform &nbsp;•&nbsp;
            Deepfake Detection · Biometric Verification · Zero-Knowledge Proofs · Blockchain Audit Trail
        </div>
        """,
        unsafe_allow_html=True,
    )
