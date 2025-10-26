
import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('./notebooks/trained_model.pkl', 'rb'))

st.title("🌾 Crop Yield Prediction App")
st.write("Predict crop yield using environmental and agricultural factors — supporting **SDG 2: Zero Hunger**")

Rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, step=1.0)
Temperature = st.number_input("🌡️ Temperature (°C)", min_value=-10.0, step=0.5)
Fertilizer_Residuals = st.number_input("🧪 Fertilizer Used (kg/ha)", min_value=0.0, step=1.0)
Soil_Moisture = st.number_input("🧱 Soil pH", min_value=0.0, max_value=14.0, step=0.1)

if st.button("🚀 Predict Yield"):
    # Example new data (raw input)
    input_data = pd.DataFrame({
    'Crop_Type': ['Rice'],
    'Disease_Presence': [0],
    'fertilizer': [100],
    'rainfall': [200],
    'soil_ph': [6.5],
    'temperature': [25]
})

# Load the encoder used during training (if you saved it)
encoder = pickle.load(open('encoder.pkl', 'rb'))

# Transform categorical features using the same encoder
encoded_data = encoder.transform(input_data[['Crop_Type', 'Disease_Presence']])

# Combine encoded categorical + numeric columns
final_input = pd.concat([encoded_data, input_data[['fertilizer', 'rainfall', 'soil_ph', 'temperature']]], axis=1)

# Predict
prediction = model.predict(final_input)
print(prediction)
st.success(f"🌾 Predicted Crop Yield: **{prediction:.2f} tons/ha**")
