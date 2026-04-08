
# 1. IMPORT LIBRARIES

import streamlit as st
import numpy as np
import joblib



# 2. LOAD SAVED OBJECTS

scaler = joblib.load('scaler.pkl')
model = joblib.load('final_employee_model.pkl')

st.set_page_config(page_title="Employee Performance Predictor", layout="centered")

# 3. TITLE

st.title("📊 Employee Performance Predictor")
st.markdown("Predict employee performance using key influencing factors.")

# 4. USER INPUTS (TOP FEATURES)

st.header("🧾 Enter Employee Details")

env_sat = st.slider("Environment Satisfaction", 1, 4, 3)
salary_hike = st.slider("Salary Hike (%)", 10, 25, 15)
years_promo = st.slider("Years Since Last Promotion", 0, 15, 2)
years_role = st.slider("Years in Current Role", 0, 20, 5)
hourly_rate = st.slider("Hourly Rate", 30, 100, 60)
years_company = st.slider("Years at Company", 0, 40, 5)
age = st.slider("Age", 18, 60, 30)
years_manager = st.slider("Years with Manager", 0, 20, 5)
distance = st.slider("Distance From Home", 1, 30, 10)
total_working_years = st.slider("Total Working Years", 0, 40, 10)

# IMPORTANT: Must match training feature order/structure
input_data = np.array([[env_sat, salary_hike, years_promo, years_role, hourly_rate, years_company, age, years_manager, distance, total_working_years]])

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

# 5. PREDICTION PIPELINE

if st.button("Predict Performance"):

    # Step 1: Scale
    input_scaled = scaler.transform(input_data)
    
    # Step 2: Predict

    prediction = model.predict(input_scaled)[0]

  
    # 6. DISPLAY RESULT
    
    st.subheader("🎯 Prediction Result")

    if prediction == 4:
        st.success("🌟 High Performer (Rating: 4)")
        st.markdown("This employee is performing at an excellent level.")

    elif prediction == 3:
        st.info("👍 Average Performer (Rating: 3)")
        st.markdown("This employee is performing well but has room for improvement.")

    else:
        st.warning("⚠️ Low Performer (Rating: 2)")
        st.markdown("This employee may require support and intervention.")


    # 7. SMART RECOMMENDATIONS
    
    st.subheader("💡 Recommendations")

    if prediction == 2:
        st.markdown("""
        - Improve work environment conditions  
        - Provide mentoring and targeted training  
        - Review salary growth and incentives  
        - Consider role realignment or support  
        """)

    elif prediction == 3:
        st.markdown("""
        - Offer career progression opportunities  
        - Encourage skill development programs  
        - Maintain good working conditions  
        - Introduce performance-based rewards  
        """)

    else:
        st.markdown("""
        - Recognize and reward high performance  
        - Provide leadership opportunities  
        - Retain through career growth plans  
        - Use as benchmark for best practices  
        """)


# 8. FOOTER

st.markdown("---")
st.markdown("📌 Data-Driven Employee Performance System")