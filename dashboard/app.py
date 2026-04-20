import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Real-Time Fraud Detection Dashboard")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    if st.button("Run Fraud Detection"):
        results = []

        for _, row in df.iterrows():
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"features": row.tolist()}
            )
            results.append(response.json())

        results_df = pd.DataFrame(results)
        final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        st.subheader("Prediction Results")
        st.dataframe(final_df)

        fraud_count = (results_df["prediction"] == 1).sum()
        high_risk_count = (results_df["risk_level"] == "high").sum()

        col1, col2 = st.columns(2)
        col1.metric("Fraudulent Transactions Detected", fraud_count)
        col2.metric("High Risk Transactions", high_risk_count)