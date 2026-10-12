import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
artifact = joblib.load("best_fraud_model_tuned.pkl")

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]
scaler = artifact["scaler"]
label_encoder = artifact["label_encoder"]
p99_amount = artifact["p99_amount"]

st.title("💳 Fraud Detection System")

st.write(
    "Enter the transaction details below to check whether "
    "the transaction is potentially fraudulent."
)

transaction_type = st.selectbox(
    "Transaction Type",
    label_encoder.classes_
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0
)

old_balance_org = st.number_input(
    "Sender Balance Before Transaction",
    min_value=0.0,
    value=1000.0
)

new_balance_orig = st.number_input(
    "Sender Balance After Transaction",
    min_value=0.0,
    value=0.0
)

old_balance_dest = st.number_input(
    "Receiver Balance Before Transaction",
    min_value=0.0,
    value=0.0
)

new_balance_dest = st.number_input(
    "Receiver Balance After Transaction",
    min_value=0.0,
    value=1000.0
)

step = st.number_input(
    "Transaction Step",
    min_value=0,
    value=1
)

if st.button("🔍 Predict"):

    # Feature engineering
    log_amount = np.log1p(amount)

    is_high_amount = int(
        amount > p99_amount
    )

    balance_diff_orig = (
        old_balance_org - new_balance_orig
    )

    balance_diff_dest = (
        new_balance_dest - old_balance_dest
    )

    hour = step % 24

    is_night = int(
        hour in [0, 1, 2, 3, 4, 5, 22, 23]
    )

    # Encode transaction type
    type_enc = label_encoder.transform(
        [transaction_type]
    )[0]

    # Create input
    input_data = pd.DataFrame([{
        "step": step,
        "type_enc": type_enc,
        "amount": amount,
        "log_amount": log_amount,
        "is_high_amount": is_high_amount,
        "balance_diff_orig": balance_diff_orig,
        "balance_diff_dest": balance_diff_dest,
        "hour": hour,
        "is_night": is_night
    }])

    # Match training feature order
    input_data = input_data[features]

    # Scale
    numeric_columns = input_data.select_dtypes(
        include=[np.number]
    ).columns

    input_data[numeric_columns] = scaler.transform(
        input_data[numeric_columns]
    )

    # Prediction
    fraud_probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = int(
        fraud_probability >= threshold
    )

    # Result
    st.subheader("Prediction Result")

    st.markdown(
        f"### Fraud Probability: {fraud_probability:.2%}"
    )

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Transaction Appears Legitimate")