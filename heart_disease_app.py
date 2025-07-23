import streamlit as st
import numpy as np
import pickle

# Load models
logistic_model = pickle.load(open("L_model", "rb"))
svm_model = pickle.load(open("svm_model", "rb"))
dt_model = pickle.load(open("DT_model", "rb"))
rf_model = pickle.load(open("RF_model", "rb"))

# Page setup
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #F8F9FA;
    }
    .stApp {
        background-color: #F8F9FA;
    }
    h1 {
        background: linear-gradient(to right, #213448, #077A7D,#456882);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 42px;
        text-align: center;
        padding-bottom: 10px;
    }
    .block-container {
        padding: 2rem 2rem;
    }
    .stButton>button {
        background-color: #560BAD;
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #7209B7;
    }
    .sidebar .sidebar-content {
        background-color: #EAE2F8;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1> Heart Disease Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("### Enter your health details below to predict your heart disease risk using different machine learning models.")

# Sidebar model selection
st.sidebar.title(" Model Selection")
model_option = st.sidebar.radio("Select Model:", (
    "Logistic Regression", "Support Vector Machine", "Decision Tree", "Random Forest"))

st.sidebar.markdown("---")
st.sidebar.write("🧠 Models are trained using real clinical data.")

# Input fields
st.markdown("---")
st.markdown("### 📝 Your Medical Information")

age = st.slider("Age", 20, 100, 45)
sex = st.radio("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.radio("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST", ["Upsloping", "Flat", "Downsloping"])
ca = st.slider("Number of Major Vessels (0–3)", 0, 3, 0)
thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# Encode inputs
sex = 1 if sex == "Male" else 0
cp = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[cp]
fbs = 1 if fbs == "Yes" else 0
restecg = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}[restecg]
exang = 1 if exang == "Yes" else 0
slope = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
thal = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}[thal]

input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                        exang, oldpeak, slope, ca, thal]])

# Prediction
if st.button("💡 Predict Risk"):
    if model_option == "Logistic Regression":
        prediction = logistic_model.predict(input_data)
    elif model_option == "Support Vector Machine":
        prediction = svm_model.predict(input_data)
    elif model_option == "Decision Tree":
        prediction = dt_model.predict(input_data)
    else:
        prediction = rf_model.predict(input_data)

    st.markdown("---")
    if prediction[0] == 1:
        st.error("🚨 **High Risk**: You might be at risk for heart disease. Please consult a healthcare provider.")
    else:
        st.success("🎉 **Low Risk**: You appear to be healthy. Keep maintaining your lifestyle!")

