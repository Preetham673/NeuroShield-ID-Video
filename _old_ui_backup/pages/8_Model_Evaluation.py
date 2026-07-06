import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.deepfake_detection.evaluate import evaluate_model

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Deepfake Detection Evaluation")

@st.cache_data
def load_results():

    return evaluate_model(show_output=False)

with st.spinner("Evaluating Model..."):

    result = load_results()

st.success("Evaluation Complete")

st.divider()

# ---------------- Metrics ----------------

c1,c2,c3,c4 = st.columns(4)

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

st.metric(
    "Inference Time",
    f"{result['inference_time']:.2f} sec"
)

st.divider()

# ---------------- Confusion Matrix ----------------

st.subheader("Confusion Matrix")

fig,ax=plt.subplots(figsize=(6,5))

sns.heatmap(

    result["confusion_matrix"],

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=["REAL","FAKE"],

    yticklabels=["REAL","FAKE"]

)

st.pyplot(fig)

st.divider()

# ---------------- ROC Curve ----------------

st.subheader("ROC Curve")

fig,ax=plt.subplots(figsize=(7,6))

ax.plot(

    result["fpr"],

    result["tpr"],

    label=f"AUC = {result['roc_auc']:.4f}"

)

ax.plot([0,1],[0,1],"--")

ax.legend()

ax.set_xlabel("False Positive Rate")

ax.set_ylabel("True Positive Rate")

st.pyplot(fig)

st.divider()

# ---------------- Classification Report ----------------

st.subheader("Classification Report")

report=pd.DataFrame(

    result["classification_report"]

).transpose()

st.dataframe(
    report,
    width="stretch"
)