
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# 1. PROJECT CONFIGURATION

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "fraud_model_improved.pkl"

# 2. STREAMLIT CONFIGURATION

st.set_page_config(
    page_title="Fraud Watch",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Fraud Watch")
st.write("Debit Card Fraud Detection System")

# 3. LOAD IMPROVED MODEL

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    fraud_model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# 4. TRANSACTION INPUT

st.subheader("Enter Transaction Details")

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0,
    step=10.0
)

use_chip = st.selectbox(
    "Payment Method",
    [
        "Swipe Transaction",
        "Chip Transaction",
        "Online Transaction"
    ]
)

mcc = st.number_input(
    "Merchant Category Code (MCC)",
    min_value=0,
    value=5411,
    step=1
)

transaction_hour = st.slider(
    "Transaction Hour",
    min_value=0,
    max_value=23,
    value=14
)

# 5. CUSTOMER TRANSACTION HISTORY

st.subheader("Customer Transaction History")

customer_prior_txn_count = st.number_input(
    "Previous Transaction Count",
    min_value=0,
    value=0,
    step=1
)

customer_prior_avg_amount = st.number_input(
    "Previous Average Transaction Amount",
    value=0.0,
    step=10.0
)

# Calculate behavioural features

if customer_prior_txn_count == 0:
    # First transaction: no previous customer history
    customer_prior_avg_amount = float("nan")
    customer_amount_deviation = float("nan")
    customer_amount_ratio_capped = float("nan")

else:
    customer_amount_deviation = (
        amount - customer_prior_avg_amount
    )

    if customer_prior_avg_amount == 0:
        customer_amount_ratio = float("nan")
    else:
        customer_amount_ratio = (
            amount / customer_prior_avg_amount
        )

    # Same limits used during model training
    customer_amount_ratio_capped = (
        max(
            -3.425680666594987,
            min(customer_amount_ratio, 7.105047307273992)
        )
        if pd.notna(customer_amount_ratio)
        else float("nan")
    )

# 6. FRAUD DETECTION

if st.button("Check Transaction"):

    transaction = pd.DataFrame([{
        "amount": amount,
        "use_chip": str(use_chip),
        "mcc": str(mcc),
        "transaction_hour": transaction_hour,
        "customer_prior_txn_count": customer_prior_txn_count,
        "customer_prior_avg_amount": customer_prior_avg_amount,
        "customer_amount_deviation": customer_amount_deviation,
        "customer_amount_ratio_capped": customer_amount_ratio_capped
    }])

    st.write("DEBUG - Features sent to model:")
    st.write(transaction.T)

    fraud_probability = fraud_model.predict_proba(
        transaction
    )[0][1]

    threshold = 0.80
    prediction = fraud_probability >= threshold

    # 7. DISPLAY RESULTS

    st.subheader("Prediction Result")

    if prediction:
        st.error("⚠️ Potential Fraud Detected")
        st.warning("This transaction has been flagged for further review.")
    else:
        st.success("✅ Transaction Appears Genuine")
        st.info("The transaction is below the current fraud alert threshold.")

    st.metric(
        "Fraud Probability",
        f"{fraud_probability:.2%}"
    )

    st.caption(f"Classification threshold: {threshold:.0%}")

    st.progress(float(fraud_probability))

# 8. MODEL INFORMATION

with st.expander("Model Information"):
    st.write("Algorithm: Logistic Regression")
    st.write("Features: 8")
    st.write("Behavioural features: Customer transaction history")
    st.write("Class weighting: Balanced")
    st.write("Experimental threshold: 0.80")
    st.write("Model: fraud_model_improved.pkl")
