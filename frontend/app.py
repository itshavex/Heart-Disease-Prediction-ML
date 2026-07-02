import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_URL = "http://127.0.0.1:8000/predict"
API_KEY = os.getenv("API_KEY", "change_this_secret")

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_home():
    st.title("❤️ Heart Disease Prediction System")
    st.markdown("""
    Welcome to the Heart Disease Prediction System!
    
    This application utilizes a state-of-the-art Machine Learning pipeline to accurately predict the likelihood of heart disease based on clinical features.
    
    👈 **Use the sidebar** to navigate between pages.
    """)
    st.image("https://via.placeholder.com/1200x400.png?text=Project+Logo+Placeholder", use_container_width=True)

def render_prediction():
    st.title("🩺 Patient Clinical Data Entry")
    
    with st.form("prediction_form"):
        st.subheader("Patient Demographics")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=55)
        with col2:
            sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
            
        st.subheader("Clinical Metrics")
        col3, col4, col5 = st.columns(3)
        with col3:
            cholst_pain = st.selectbox("Chest Pain Type", options=[("Typical Angina", 0), ("Atypical Angina", 1), ("Non-anginal", 2), ("Asymptomatic", 3)], format_func=lambda x: x[0])[1]
            rest_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=140)
            chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=250)
            
        with col4:
            fast_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
            rest_ecg = st.selectbox("Resting ECG Results", options=[("Normal", 0), ("ST-T Wave Abnormality", 1), ("Left Ventricular Hypertrophy", 2)], format_func=lambda x: x[0])[1]
            thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=220, value=150)
            
        with col5:
            exang = st.selectbox("Exercise Induced Angina", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
            oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
            slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)], format_func=lambda x: x[0])[1]
            
        submit = st.form_submit_button("Predict Heart Disease", type="primary", use_container_width=True)
        
    if submit:
        payload = {
            "age": age,
            "sex": sex,
            "cholst pain": cholst_pain,
            "RestBP": rest_bp,
            "CholL": chol,
            "FastBS": fast_bs,
            "RESTECG": rest_ecg,
            "Thalach": thalach,
            "EXAng": exang,
            "OLDPeak": oldpeak,
            "SLOPE": slope
        }
        
        with st.spinner("Analyzing patient metrics against the Machine Learning model..."):
            try:
                headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
                response = requests.post(API_URL, json=payload, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state['last_prediction'] = data
                    st.success("Prediction generated successfully! Navigate to the 'Results' page.")
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to the FastAPI backend: {e}")

def render_results():
    st.title("📊 Prediction Results")
    
    if 'last_prediction' not in st.session_state:
        st.info("No prediction data available. Please submit a prediction from the 'Prediction' page first.")
        return
        
    data = st.session_state['last_prediction']
    
    if data.get("prediction") == 1:
        st.error("🚨 **HIGH RISK**: The model predicts a high likelihood of Heart Disease.")
    else:
        st.success("✅ **LOW RISK**: The model predicts a low likelihood of Heart Disease.")
        
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Probability of Disease", value=f"{data.get('probability', 0) * 100:.2f}%")
    with col2:
        st.metric(label="Model Confidence", value="High")
    with col3:
        st.metric(label="API Status", value=data.get("status", "success").upper())
    
    st.warning("Note: The FastAPI backend currently does not emit SHAP values natively for the UI. (Placeholder for future Explainability metrics).")

def render_about():
    st.title("ℹ️ About the Project")
    st.markdown("""
    This frontend is built using **Streamlit** to interface with the Research-Grade **FastAPI** Machine Learning Backend.
    
    ### Architecture
    - **Frontend UI**: Streamlit (Python)
    - **Backend API**: FastAPI (Python)
    - **Machine Learning**: Scikit-Learn, XGBoost, Pandas
    
    Developed as part of an end-to-end Machine Learning pipeline to demonstrate production deployment patterns.
    """)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Results", "About"])

if page == "Home":
    render_home()
elif page == "Prediction":
    render_prediction()
elif page == "Results":
    render_results()
elif page == "About":
    render_about()
