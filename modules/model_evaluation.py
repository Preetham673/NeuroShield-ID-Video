import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.deepfake_detection.evaluate import evaluate_model

from components.ui import page_header, section_divider, stat_card, skeleton

PLOT_TEMPLATE = "plotly_dark"


@st.cache_data
def load_results():
    return evaluate_model(show_output=False)


def _style_fig(fig):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#CBD5E1"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def render():

    page_header("📈", "Model Evaluation", "Deepfake detection model performance, diagnostics, and reliability metrics.")

    placeholder = st.empty()
    with placeholder.container():
        skeleton(height_px=100, count=4)

    result = load_results()
    placeholder.empty()

    st.success("Evaluation complete.")
    section_divider()

    # ---------------- Metrics ----------------
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        stat_card("🎯", "Accuracy", f"{result['accuracy']*100:.2f}%", 1)
    with c2:
        stat_card("📐", "Precision", f"{result['precision']*100:.2f}%", 2)
    with c3:
        stat_card("🔁", "Recall", f"{result['recall']*100:.2f}%", 3)
    with c4:
        stat_card("⚖️", "F1 Score", f"{result['f1']*100:.2f}%", 4)
    with c5:
        stat_card("⚡", "Inference", f"{result['inference_time']:.2f}s", 5)

    section_divider()

    # ---------------- Confusion Matrix + ROC ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix")
        cm = result["confusion_matrix"]
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["REAL", "FAKE"],
                y=["REAL", "FAKE"],
                colorscale="Greens",
                text=cm,
                texttemplate="%{text}",
                showscale=False,
            )
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(_style_fig(fig), use_container_width=True)

    with col2:
        st.markdown("#### ROC Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result["fpr"], y=result["tpr"], mode="lines",
            name=f"AUC = {result['roc_auc']:.4f}",
            line=dict(color="#00E676", width=3),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#64748B", dash="dash"), showlegend=False,
        ))
        fig.update_xaxes(title="False Positive Rate")
        fig.update_yaxes(title="True Positive Rate")
        st.plotly_chart(_style_fig(fig), use_container_width=True)

    section_divider()

    # ---------------- Classification Report ----------------
    st.markdown("#### Classification Report")
    report = pd.DataFrame(result["classification_report"]).transpose()
    st.dataframe(report, use_container_width=True)

    section_divider()

    # ---------------- Model / Dataset Info ----------------
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Model Information")
        st.info(
            """
            **Architecture:** EfficientNet-B0

            **Task:** Binary Deepfake Classification

            **Output Classes:** REAL / FAKE
            """
        )
    with col4:
        st.markdown("#### Dataset Information")
        st.info(
            f"""
            **Test Set Size:** {result['num_samples']:,} images

            **Inference Time:** {result['inference_time']:.2f} seconds

            **Evaluation Mode:** Held-out test split
            """
        )