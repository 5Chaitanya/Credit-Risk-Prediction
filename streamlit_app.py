import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load model and scaler
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("credit_risk_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

st.title("💳 Credit Risk Prediction App (Streamlit Only)")
st.markdown("Enter loan applicant details below:")

# -- Input fields --
age = st.slider("Person Age", 18, 75, 30)
income = st.number_input("Annual Income ($)", min_value=5000, max_value=300000, value=50000)
emp_length = st.slider("Employment Length (Years)", 0, 50, 5)

home_options = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
home_ownership = st.selectbox("Home Ownership", list(home_options.keys()))

intent_options = {"PERSONAL": 0, "EDUCATION": 1, "MEDICAL": 2,
                  "VENTURE": 3, "HOMEIMPROVEMENT": 4, "DEBTCONSOLIDATION": 5}
loan_intent = st.selectbox("Loan Intent", list(intent_options.keys()))

grade_options = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
loan_grade = st.selectbox("Loan Grade", list(grade_options.keys()))

loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, value=10000)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=35.0, value=12.5)

default_file = st.selectbox("Previous Default on File?", ["Yes", "No"])
default_on_file = 1 if default_file == "Yes" else 0

cred_hist = st.slider("Credit History Length (Years)", 0, 30, 5)

# Derived feature
loan_percent_income = loan_amnt / (income)

# Feature list
features = [age, income, emp_length,
            home_options[home_ownership],
            intent_options[loan_intent],
            grade_options[loan_grade],
            loan_amnt, loan_int_rate,
            loan_percent_income, default_on_file,
            cred_hist]

# Prediction
if st.button("🧠 Predict Credit Risk"):
    # Reshape + Scale
    input_array = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prob = model.predict(input_scaled)[0][0]
    prediction = int(prob >= 0.5)

    st.subheader("📊 Prediction Result")
    st.success("✅ No Default Risk" if prediction == 0 else "❌ High Default Risk")
    st.metric("Probability of Default", f"{prob*100:.2f}%")
