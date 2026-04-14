import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Try to import plotly, or show a helpful error if missing
try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    st.error("Error: 'plotly' is not installed. Please add 'plotly' to your requirements.txt file.")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        with open("best_churn_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("❌ 'best_churn_model.pkl' not found in the repository.")
        return None

model = load_model()

if model:
    st.success("✅ Model loaded successfully!")

    # ---------------- UI LAYOUT ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Customer Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with col2:
        st.subheader("💳 Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

    # ---------------- PREDICTION ----------------
    if st.button("🔮 Predict Churn", type="primary"):
        # Create input dataframe
        input_data = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": 1 if partner == "Yes" else 0,
            "Dependents": 1 if dependents == "Yes" else 0,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges
        }

        input_df = pd.DataFrame([input_data])

        # One-hot encoding to match training format
        input_encoded = pd.get_dummies(input_df)

        # Align with model features (Fill missing columns with 0)
        if hasattr(model, 'feature_names_in_'):
            input_encoded = input_encoded.reindex(
                columns=model.feature_names_in_,
                fill_value=0
            )

        # Prediction logic
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0]
        churn_prob = probability[1] * 100

        # ---------------- OUTPUT ----------------
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.subheader("📌 Result")
            if prediction == 1:
                st.error("🚨 HIGH RISK: Customer likely to churn")
                st.metric("Churn Probability", f"{churn_prob:.1f}%")
            else:
                st.success("✅ LOW RISK: Customer likely to stay")
                st.metric("Retention Probability", f"{100 - churn_prob:.1f}%")

        with res_col2:
            # Visualization
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = churn_prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Risk Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if churn_prob > 50 else "green"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
