import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load saved models in the Keras format
mlp_model = load_model('/mnt/data/mlp_gdp_model.keras')
rnn_model = load_model('/mnt/data/rnn_gdp_model.keras')


# Scaler for consistent input transformation
scaler = MinMaxScaler(feature_range=(0, 1))

st.title("GDP Prediction App")
st.write("Predict future GDP values using either MLP or RNN models based on previous years' data.")

# User input for previous 3 years' data
st.write("Enter data for the last 3 years to predict the next year's Nominal GDP price.")
input_data = []
for i in range(3):
    year_data = st.number_input(f"Year {i+1} Nominal GDP prices (Ksh Million)", min_value=0.0)
    growth_rate = st.number_input(f"Year {i+1} Annual GDP growth (%)")
    real_gdp = st.number_input(f"Year {i+1} Real GDP prices (Ksh Million)", min_value=0.0)
    input_data.append([year_data, growth_rate, real_gdp])

# Convert and scale input data
input_data = np.array(input_data).reshape(1, 3, 3)
input_data_scaled = scaler.fit_transform(input_data.reshape(-1, 3)).reshape(1, 3, 3)

# Model Selection
model_choice = st.selectbox("Select Model", ["MLP", "RNN"])

if st.button("Predict"):
    if model_choice == "MLP":
        # Flatten input for MLP model
        input_data_flattened = input_data_scaled.reshape(1, -1)
        prediction = mlp_model.predict(input_data_flattened)
    else:
        prediction = rnn_model.predict(input_data_scaled)
    
    # Display prediction result
    st.write(f"Predicted Nominal GDP price for the next year: {prediction[0][0]:.2f} Ksh Million")
