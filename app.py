import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the saved models
mlp_model = load_model('mlp_model.h5')
rnn_model = load_model('rnn_model.h5')

# Streamlit app
st.title("GDP Prediction App")

# Input for new data
st.write("Enter GDP-related data:")
year = st.number_input("Year", min_value=2000, max_value=2100, value=2023)
nominal_gdp = st.number_input("Nominal GDP prices (Ksh Million)", min_value=0.0, value=0.0)
annual_growth = st.number_input("Annual GDP growth (%)", value=0.0)
real_gdp = st.number_input("Real GDP prices (Ksh Million)", min_value=0.0, value=0.0)

# Button to preprocess and make predictions
if st.button("Predict"):
    try:
        # Preprocess inputs
        input_data = np.array([[nominal_gdp, annual_growth, real_gdp]])
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_input = scaler.fit_transform(input_data)

        # Prepare sequence (mock sequence for demonstration)
        sequence = np.array([scaled_input, scaled_input, scaled_input])
        sequence = sequence.reshape(1, 3, 3)

        # MLP prediction
        mlp_prediction = mlp_model.predict(sequence)
        st.write(f"MLP Prediction (Nominal GDP): {mlp_prediction[0][0]}")

        # RNN prediction
        rnn_prediction = rnn_model.predict(sequence)
        st.write(f"RNN Prediction (Nominal GDP): {rnn_prediction[0][0]}")

    except Exception as e:
        st.error(f"Error: {e}")
