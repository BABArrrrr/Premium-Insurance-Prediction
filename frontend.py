import streamlit as st
import requests

API_URL = 'http://127.0.0.1:8000/predict'

st.title('Insurance Premium Category Predictor')

st.markdown('Enter Your details here')

age = st.number_input('age...',min_value=0, max_value=120, value=30)
weight = st.number_input('weight (kg)...',min_value=1.0, value=65.0)
height = st.number_input('height (m)...',min_value=0.5, max_value=2.5,value=1.7)
income_lpa = st.number_input("Annual Income (LPA)", min_value=0.1, value=10.0)
smoker = st.selectbox("Are you a smoker?", options=[True, False])
city = st.text_input("City", value="Mumbai")
occupation = st.selectbox(
    "Occupation",
    ['retired', 'freelancer', 'student', 'government_job', 'business_owner', 'unemployed', 'private_job']
)

if st.button('Predict Category'):

    input_data = {
        'age':age,
        'weight': weight,
        'height':height,
        'income_lpa': income_lpa,
        'smoker': smoker,
        'city':city,
        'occupation':occupation
    }

    try:
        response= requests.post(API_URL,json=input_data)
        if response.status_code==200:
            result = response.json()
            st.success(f'Predict Insurance Premium Category: **{result['predicted_category']}**')            
        else:
            st.error(f'API Error: {response.status_code} - {response.text}')        
    except requests.exceptions.ConnectionError:
        st.error('Could not connect to the FastAPI server. Make sure it is running on port 8000.')        