
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# Load saved model files
# -----------------------------

import joblib
import xgboost as xgb

@st.cache_resource
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model("xgboost_fraud_model.json")

    scaler = joblib.load("feature_scaler.pkl")
    columns = joblib.load("feature_columns.pkl")

    return model, scaler, columns

model, scaler, feature_columns = load_artifacts()

# -----------------------------
# Dashboard heading
# -----------------------------
st.title("🛡️ Credit Card Fraud Detection")
st.write(
    "An ML-based demonstration that estimates "
    "whether a transaction may be fraudulent."
)

st.info(
    "Educational project only. Predictions are estimates "
    "and must not be treated as confirmed fraud."
)

# -----------------------------
# Input form
# -----------------------------
st.subheader("Enter Transaction Details")

with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        time_value = st.number_input(
            "Time (seconds from dataset start)",
            min_value=0.0,
            value=0.0
        )

    with col2:
        amount_value = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0
        )

    st.caption(
        "Enter the anonymized V1–V28 features from a dataset "
        "record. Arbitrary values may not represent a realistic "
        "transaction."
    )

    feature_values = {}

    for i, column in enumerate(
        [c for c in feature_columns if c.startswith("V")],
        start=1
    ):
        feature_values[column] = st.number_input(
            column,
            value=0.0,
            format="%.6f",
            key=f"feature_{column}"
        )

    submitted = st.form_submit_button(
        "Analyze Transaction",
        type="primary"
    )

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    input_data = {
        "Time": time_value,
        "Amount": amount_value,
        **feature_values
    }

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    # Apply the same scaling used during training
    input_df[["Time", "Amount"]] = scaler.transform(
        input_df[["Time", "Amount"]]
    )

    # Predict class and probability
    prediction = int(model.predict(input_df)[0])
    fraud_probability = float(
        model.predict_proba(input_df)[0][1]
    )

    st.divider()
    st.subheader("Prediction Results")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            "Estimated Fraud Probability",
            f"{fraud_probability:.2%}"
        )

    with metric2:
        st.metric(
            "Predicted Class",
            "Potential Fraud" if prediction == 1
            else "Likely Legitimate"
        )

    if prediction == 1:
        st.error(
            "⚠️ The model has flagged this transaction "
            "as potentially fraudulent."
        )
    else:
        st.success(
            "The model has classified this transaction "
            "as likely legitimate."
        )

    st.caption(
        "The probability is a model estimate, not a guarantee "
        "that the transaction is or is not fraudulent."
    )

# -----------------------------
# Model Performance
# -----------------------------
st.divider()

st.header("📊 Model Performance")
st.write("XGBoost evaluation results on the held-out test dataset.")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Precision", "84.44%")
col2.metric("Recall", "80.00%")
col3.metric("F1-Score", "0.8216")
col4.metric("ROC-AUC", "0.9714")

st.caption(
    "These are the previously measured test-set results, "
    "not performance estimates for the individual transaction above."
)



# -----------------------------
# Confusion Matrix - Fixed Version
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

st.subheader("Confusion Matrix")

cm = np.array([
    [56637, 14],
    [19, 76]
])

fig, ax = plt.subplots(figsize=(6, 4))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("XGBoost Confusion Matrix")
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# -----------------------------
# Model Comparison Chart
# -----------------------------
st.divider()
st.subheader("📊 Model Comparison")

comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "XGBoost"],
    "Precision": [0.0562, 0.8444],
    "Recall": [0.8737, 0.8000],
    "F1-score": [0.1057, 0.8216]
})

st.write(
    "Compare the performance of both models "
    "on the held-out test dataset."
)

comparison_long = comparison_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

chart = comparison_long.pivot(
    index="Metric",
    columns="Model",
    values="Score"
)

st.bar_chart(
    chart,
    y_label="Score",
    stack=False
)

st.dataframe(
    comparison_df.set_index("Model").style.format({
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1-score": "{:.4f}"
    })
)
