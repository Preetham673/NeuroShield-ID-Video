import streamlit as st
import matplotlib.pyplot as plt

from src.deepfake_detection.evaluate import evaluate_model

st.set_page_config(
    page_title="Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 NeuroShield-ID Analytics Dashboard")

@st.cache_data
def load_metrics():
    return evaluate_model(show_output=False)

with st.spinner("Loading analytics..."):
    result = load_metrics()

st.success("Analytics Loaded Successfully")

st.divider()

# ===================================================
# KPI CARDS
# ===================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Accuracy",
    f"{result['accuracy']*100:.2f}%"
)

c2.metric(
    "Precision",
    f"{result['precision']*100:.2f}%"
)

c3.metric(
    "Recall",
    f"{result['recall']*100:.2f}%"
)

c4.metric(
    "F1 Score",
    f"{result['f1']*100:.2f}%"
)

st.divider()

# ===================================================
# PIE CHART
# ===================================================

st.subheader("Prediction Distribution")

cm = result["confusion_matrix"]

real_predictions = cm[0][0] + cm[1][0]
fake_predictions = cm[0][1] + cm[1][1]

fig, ax = plt.subplots(figsize=(6,6))

ax.pie(
    [real_predictions, fake_predictions],
    labels=["REAL", "FAKE"],
    autopct="%1.1f%%",
    startangle=90
)

ax.axis("equal")

st.pyplot(fig)

st.divider()

# ===================================================
# BAR CHART
# ===================================================

st.subheader("Model Performance")

metrics = [
    result["accuracy"] * 100,
    result["precision"] * 100,
    result["recall"] * 100,
    result["f1"] * 100
]

labels = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

fig, ax = plt.subplots(figsize=(8,5))

ax.bar(labels, metrics)

ax.set_ylim(0,100)

ax.set_ylabel("Score (%)")

for i, value in enumerate(metrics):
    ax.text(
        i,
        value + 1,
        f"{value:.2f}",
        ha="center"
    )

st.pyplot(fig)

st.divider()

# ===================================================
# MODEL SUMMARY
# ===================================================

st.subheader("Model Summary")

st.info(f"""
Model Architecture : EfficientNet-B0

Dataset : Processed Data Subset

Test Images : 10,000

Inference Time : {result['inference_time']:.2f} seconds

Deepfake Classes : REAL / FAKE
""")