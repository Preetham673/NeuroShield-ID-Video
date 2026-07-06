"""
components/ui.py

Reusable, purely-presentational UI building blocks for NeuroShield-ID.
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
        <div class="ns-fade">
            <h1 style="margin-bottom:2px;">{icon} {title}</h1>
            <p style="margin-top:0;color:var(--text-mid);font-size:1rem;">{subtitle}</p>
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


_ICON_COLORS = ["ns-icon-blue", "ns-icon-violet", "ns-icon-cyan", "ns-icon-pink", "ns-icon-amber", "ns-icon-emerald"]


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
        <h4 style="margin:4px 0;">{title}</h4>
        <p style="font-size:.88rem;margin:0;">{desc}</p>
    </div>
    """
    st.markdown(fade(html, delay), unsafe_allow_html=True)


def code_card(label: str, value: str):
    st.markdown(
        f"""
        <p style="font-size:.82rem;color:var(--text-lo);margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em;">{label}</p>
        <div class="ns-code-card">{value}</div>
        """,
        unsafe_allow_html=True,
    )


def circular_gauge(value: float, label: str = "Trust Score", max_value: float = 100.0):
    """
    Renders a conic-gradient circular gauge purely in HTML/CSS (no extra deps).
    value is expected 0-100 (or 0-max_value).
    """
    pct = max(0.0, min(1.0, value / max_value))
    deg = pct * 360

    if pct >= 0.75:
        color = "#22c55e"
    elif pct >= 0.5:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    html = f"""
    <div class="ns-gauge-wrap">
        <div class="ns-gauge" style="background: conic-gradient({color} {deg}deg, rgba(255,255,255,0.08) {deg}deg 360deg);">
            <div class="ns-gauge-inner">
                <div class="ns-gauge-value">{value:.1f}</div>
                <div class="ns-gauge-label">{label}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def labeled_progress(label: str, value: float, max_value: float = 100.0):
    pct = max(0.0, min(1.0, value / max_value))
    st.markdown(
        f'<p style="font-size:.85rem;margin-bottom:4px;color:var(--text-mid);">{label} '
        f'<span style="float:right;color:var(--text-hi);font-weight:600;">{value:.2f}%</span></p>',
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
            🛡️ NeuroShield-ID &nbsp;•&nbsp; AI-Powered Secure Identity Verification &nbsp;•&nbsp;
            Deepfake Detection · ZKP · Blockchain · Explainable AI
        </div>
        """,
        unsafe_allow_html=True,
    )
