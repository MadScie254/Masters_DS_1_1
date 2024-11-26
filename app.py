import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the models
mlp_model = load_model('mlp_model.h5')
rnn_model = load_model('rnn_model.h5')

# Streamlit app
st.title("GDP Prediction App")

# Input for new data
st.write("Enter GDP data:")
nominal_gdp = st.number_input("Nominal GDP prices (Ksh Million)", min_value=0.0, value=0.0)
annual_growth = st.number_input("Annual GDP growth (%)", value=0.0)
real_gdp = st.number_input("Real GDP prices (Ksh Million)", min_value=0.0, value=0.0)

# Prepare input array
input_data = np.array([[nominal_gdp, annual_growth, real_gdp]])
sequence_length = 3
sequence = np.tile(input_data, (sequence_length, 1))  # Repeat the input to form a sequence
sequence = sequence.reshape(1, sequence_length, 3)  # Shape (1, 3, 3)

# Prediction buttons
if st.button("Predict with MLP"):
    try:
        mlp_prediction = mlp_model.predict(sequence)
        st.success(f"MLP Model Prediction (Nominal GDP): {mlp_prediction[0][0]}")
    except Exception as e:
        st.error(f"Error in MLP Prediction: {e}")

if st.button("Predict with RNN"):
    try:
        rnn_prediction = rnn_model.predict(sequence)
        st.success(f"RNN Model Prediction (Nominal GDP): {rnn_prediction[0][0]}")
    except Exception as e:
        st.error(f"Error in RNN Prediction: {e}")
