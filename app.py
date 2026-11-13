from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="centered")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "fraud_model.joblib"

TXN_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        files = sorted(p.name for p in BASE_DIR.iterdir())
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH.name}. Files in this folder: {files}"
        )
    return joblib.load(MODEL_PATH)


def prepare_input(user_inputs: dict, model) -> pd.DataFrame:
    """Notebook jaisi hi preprocessing: engineered features + one-hot + column order."""
    row = pd.DataFrame([user_inputs])

    row["origin_error"] = (row["oldbalanceOrg"] - row["amount"]) - row["newbalanceOrig"]
    row["dest_error"] = (row["oldbalanceDest"] + row["amount"]) - row["newbalanceDest"]

    row = pd.get_dummies(row, columns=["type"], dtype=int)

    # Model ke exact columns aur exact order; jo column na ho wo 0
    return row.reindex(columns=model.feature_names_in_, fill_value=0)


# ---------------- UI ----------------
st.title("🛡️ Mobile Transactions Fraud Detection")
st.caption("Random Forest model | Enter the transaction details and check for fraud risk.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the model file: {e}")
    st.stop()

with st.sidebar:
    st.header("Prediction Settings")
    threshold = st.slider(
        "Probability threshold",
        0.05, 0.95, 0.50, 0.05,
        help="If the fraud probability exceeds this, the transaction will be considered fraudulent.",
    )
    st.divider()
    st.subheader("About the Model")
    st.write("Random Forest Fraud Classifier")
    st.caption(f"Input features: {len(model.feature_names_in_)}")

with st.form("txn_form"):
    txn_type = st.selectbox("Transaction type", TXN_TYPES, index=4)
    amount = st.number_input("Amount", min_value=0.0, value=181.0, step=100.0, format="%.2f")

    c1, c2 = st.columns(2)
    with c1:
        oldbalanceOrg = st.number_input("Sender balance (before)", min_value=0.0, value=181.0, format="%.2f")
        newbalanceOrig = st.number_input("Sender balance (after)", min_value=0.0, value=0.0, format="%.2f")
    with c2:
        oldbalanceDest = st.number_input("Receiver balance (before)", min_value=0.0, value=0.0, format="%.2f")
        newbalanceDest = st.number_input("Receiver balance (after)", min_value=0.0, value=0.0, format="%.2f")

    submitted = st.form_submit_button("Predict Fraud Risk")

if submitted:
    user_inputs = {
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "type": txn_type,
    }

    try:
        X_new = prepare_input(user_inputs, model)
        proba = float(model.predict_proba(X_new)[0, 1])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("Result")
    st.metric("Fraud probability", f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))

    if proba >= threshold:
        st.error("⚠️ High fraud risk: This transaction is suspicious.")
    else:
        st.success("✅ Low fraud risk: This transaction appears normal.")

    with st.expander("The input was sent to the model."):
        st.dataframe(X_new)
