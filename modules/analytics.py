import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.deepfake_detection.evaluate import evaluate_model

from components.ui import page_header, section_divider, stat_card, skeleton

PLOT_TEMPLATE = "plotly_dark"
BRAND_COLORS = ["#00E676", "#00FF88", "#00A854", "#22C55E", "#FFB020"]


@st.cache_data
def load_metrics():
    return evaluate_model(show_output=False)


def _style_fig(fig):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#CBD5E1"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def render():

    page_header("📊", "Analytics Dashboard", "Model performance and verification insights at a glance.")

    placeholder = st.empty()
    with placeholder.container():
        skeleton(height_px=100, count=4)

    result = load_metrics()
    placeholder.empty()

    st.success("Analytics loaded successfully.")
    section_divider()

    # ---------------- KPI Cards ----------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("🎯", "Accuracy", f"{result['accuracy']*100:.2f}%", 1)
    with c2:
        stat_card("📐", "Precision", f"{result['precision']*100:.2f}%", 2)
    with c3:
        stat_card("🔁", "Recall", f"{result['recall']*100:.2f}%", 3)
    with c4:
        stat_card("⚖️", "F1 Score", f"{result['f1']*100:.2f}%", 4)

    section_divider()

    cm = result["confusion_matrix"]
    real_predictions = cm[0][0] + cm[1][0]
    fake_predictions = cm[0][1] + cm[1][1]

    # ---------------- Pie + Bar ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Prediction Distribution")
        fig = go.Figure(
            data=[go.Pie(
                labels=["REAL", "FAKE"],
                values=[real_predictions, fake_predictions],
                hole=0.55,
                marker=dict(colors=["#00E676", "#FF4D5A"]),
            )]
        )
        st.plotly_chart(_style_fig(fig), use_container_width=True)

    with col2:
        st.markdown("#### Model Performance")
        labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
        metrics = [
            result["accuracy"] * 100,
            result["precision"] * 100,
            result["recall"] * 100,
            result["f1"] * 100,
        ]
        perf_colors = ["#00E676", "#00D2FF", "#3B82F6", "#FFB020"]
        fig = go.Figure(
            data=[go.Bar(
                x=labels,
                y=metrics,
                marker_color=perf_colors,
                text=[f"{v:.2f}%" for v in metrics],
                textposition="outside"
            )]
        )
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(_style_fig(fig), use_container_width=True)

    section_divider()

    # ---------------- Line + Area (trend illustration from available metrics) ----------------
    col3, col4 = st.columns(2)

    trend_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score": [result["accuracy"] * 100, result["precision"] * 100, result["recall"] * 100, result["f1"] * 100],
    })

    with col3:
        st.markdown("#### Score Trend")
        fig = go.Figure(
            data=[go.Scatter(
                x=trend_df["Metric"],
                y=trend_df["Score"],
                mode="lines+markers",
                line=dict(color="#00D2FF", width=3),
                marker=dict(size=9, color="#00E676")
            )]
        )
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(_style_fig(fig), use_container_width=True)

    with col4:
        st.markdown("#### Cumulative Confidence Area")
        fig = go.Figure(
            data=[go.Scatter(
                x=trend_df["Metric"],
                y=trend_df["Score"],
                fill="tozeroy",
                mode="lines",
                line=dict(color="#00E676", width=2),
                fillcolor="rgba(0, 230, 118, 0.18)"
            )]
        )
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(_style_fig(fig), use_container_width=True)

    section_divider()

    # ---------------- Model Summary ----------------
    st.markdown("#### Model Summary")
    st.info(
        f"""
        **Model Architecture:** EfficientNet-B0

        **Dataset:** Processed Data Subset

        **Test Images:** {result['num_samples']:,}

        **Inference Time:** {result['inference_time']:.2f} seconds

        **Deepfake Classes:** REAL / FAKE
        """
    )