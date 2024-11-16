import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load Models
mlp_model = load_model("mlp_gdp_model.keras")
rnn_model = load_model("rnn_gdp_model.keras")

# Input for testing
st.title("GDP Prediction with Neural Networks")
st.write("Input 3x3 GDP-related data for predictions:")

input_data = []
for i in range(3):
    row = st.text_input(f"Row {i+1} (comma-separated):", "0.0, 0.0, 0.0")
    input_data.append(list(map(float, row.split(','))))

if st.button("Predict"):
    # Reshape and preprocess the input
    input_array = np.array(input_data).reshape(1, 3, 3)

    # Predictions
    mlp_prediction = mlp_model.predict(input_array)[0][0]
    rnn_prediction = rnn_model.predict(input_array)[0][0]

    # Display Results
    st.write(f"MLP Prediction: {mlp_prediction}")
    st.write(f"RNN Prediction: {rnn_prediction}")
