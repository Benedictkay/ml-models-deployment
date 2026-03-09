import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Get the directory where app.py is located
curr_dir = Path(__file__).parent

model = joblib.load("ridge_patient_footfall_model.pkl")
scaler = joblib.load("ridge_scaler.pkl")
model_features = joblib.load("model_features.pkl")

st.title("Healthcare Facility Footfall Predictor")

zone_no = st.selectbox("Zone Number", [1,2,3,4,5])
beds_emergency = st.slider("Emergency Beds", 0, 100, 10)
total_beds = st.slider("Total Beds", 0, 500, 20)
doctors = st.number_input("Doctors", 0, 100, 5)
nurses = st.number_input("Nurses", 0, 200, 10)
midwives = st.number_input("Midwives", 0, 50, 2)
ambulances = st.number_input("Ambulances", 0, 10, 1)

if st.button("Predict Footfall"):

    total_staff = doctors + nurses + midwives
    log_beds = np.log1p(total_beds)

    input_data = pd.DataFrame({
        "Zone No.":[zone_no],
        "Number of Beds in Emergency Wards":[beds_emergency],
        "Total_Staff":[total_staff],
        "Log_Beds":[log_beds],
        "Count of Ambulance":[ambulances]
    })

    # Create empty dataframe with training features
    input_df = pd.DataFrame(columns=model_features)
    input_df.loc[0] = 0

    # Fill available features
    for col in input_data.columns:
        if col in input_df.columns:
            input_df[col] = input_data[col]

    # Scale
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.success(f"Estimated Monthly Patient Footfall: {int(prediction)}")