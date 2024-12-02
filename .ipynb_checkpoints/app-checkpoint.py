import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load the trained model
model = load_model('gdp_prediction_model.keras')

# Streamlit app
def main():
    st.title("GDP Prediction Model")
    st.write("This app predicts the upcoming GDP based on prior GDP data.")

    # User input for prior GDP data
    st.subheader("Input Prior GDP Data")
    
    # Define fields for prior GDP data (example: 10 data points)
    prior_gdp_data = []
    for i in range(1, 11):
        gdp_input = st.number_input(f"GDP Data Point {i}", min_value=0.0, step=0.1)
        prior_gdp_data.append(gdp_input)

    # When the user clicks 'Predict'
    if st.button("Predict"):
        # Check if all input data is entered
        if len(prior_gdp_data) == 10:
            # Preprocess the input data
            input_data = np.array(prior_gdp_data).reshape(1, 10, 1)  # Reshaping for the LSTM model
            scaler = MinMaxScaler(feature_range=(0, 1))  # For scaling the GDP values
            input_data_scaled = scaler.fit_transform(input_data[0]).reshape(1, 10, 1)

            # Make a prediction using the model
            prediction = model.predict(input_data_scaled)
            predicted_gdp = prediction[0][0]  # Extract predicted GDP

            st.subheader(f"Predicted Upcoming GDP: {predicted_gdp:.2f}")
        else:
            st.error("Please provide all 10 prior GDP data points.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
