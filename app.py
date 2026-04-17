import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title='Customer Churn Predictor',
    page_icon='📊',
    layout='wide'
)

# Title
st.title('Customer Churn Prediction System')

# Load model
@st.cache_resource
def load_model():
    with open('best_churn_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()
st.success('Model loaded successfully!')

# Layout
col1, col2 = st.columns(2)

# Column 1
with col1:
    st.subheader('Customer Demographics')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['No', 'Yes'])
    dependents = st.selectbox('Dependents', ['No', 'Yes'])

# Column 2
with col2:
    st.subheader('Account Information')
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    monthly_charges = st.number_input(
        'Monthly Charges ($)',
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

# Prediction button
# Prediction button
if st.button('Predict Churn', type='primary'):

    # Create input dataframe
    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges
    }

    input_df = pd.DataFrame([input_data])

    # Encode input
    input_encoded = pd.get_dummies(input_df)

    # Align columns with model
    try:
        model_columns = model.feature_names_in_
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    except:
        pass

    # Prediction
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0]
    churn_prob = probability[1] * 100
    retention_prob = 100 - churn_prob

    # Layout for results
    col3, col4 = st.columns(2)

    # 🔹 Metric Display
    with col3:
        if prediction == 1:
            st.error('HIGH RISK: Customer likely to churn')
        else:
            st.success('LOW RISK: Customer likely to stay')

        st.metric("Churn Probability", f"{churn_prob:.1f}%")
        st.metric("Retention Probability", f"{retention_prob:.1f}%")

    # 🔹 Gauge Chart
    with col4:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob,
            title={'text': "Churn Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if churn_prob > 50 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)
