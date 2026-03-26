import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Yashaswinishri/Tourism-Package-Prediction", filename="toursim_pred_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package
st.title("Tourism Package - Customer Purchase Predictor App")
st.write("Tourism Package - Customer Purchase Predictor App is an internal tool for Vist with US Marketing staff that predicts whether a customer will purchase a product or not based on the details.")
st.write("Kindly enter the customer details to check whether they are likely to churn.")

# Collect user input
st.set_page_config(page_title="Tourism Package - Customer Purchase Predictor App", page_icon="📊", layout="wide")

# ---------- Custom Styling ----------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.title("🏡 Travel Package Prediction Form")
st.markdown("Fill in the details below to predict customer interest.")

# ---------- Sidebar ----------
st.sidebar.header("📌 Navigation")
st.sidebar.info("Provide customer details to generate prediction.")

# ---------- Form ----------
with st.form("prediction_form"):

    st.subheader("👤 Personal Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married","UnMarried" "Divorced"])

    with col2:
        occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer","Large Business", "Small Business"])
        designation = st.selectbox("Designation", ["Designation","Manager","Executive","Senior Manager","AVP","VP"])
        monthly_income = st.number_input("Monthly Income", min_value=0)

    with col3:
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        own_car = st.selectbox("Own Car", [0, 1])
        passport = st.selectbox("Passport", [0, 1])

    st.divider()

    st.subheader("📞 Interaction Details")
    col4, col5, col6 = st.columns(3)

    with col4:
        typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        duration_pitch = st.slider("Duration of Pitch (mins)", 0, 500)

    with col5:
        number_of_followups = st.number_input("Number of Followups", min_value=0)
        pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5)

    with col6:
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard","King", "Deluxe", "Super Deluxe"])
        preferred_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])

    st.divider()

    st.subheader("👨‍👩‍👧 Travel Details")
    col7, col8, col9 = st.columns(3)

    with col7:
        number_of_persons = st.number_input("Number Of Persons Visiting", min_value=1)

    with col8:
        number_of_children = st.number_input("Number Of Children Visiting", min_value=0)

    with col9:
        number_of_trips = st.number_input("Number Of Trips", min_value=0)

    # ---------- Submit ----------
    submitted = st.form_submit_button("🚀 Predict")

# ---------- Output ----------
if submitted:
    st.success("✅ Data submitted successfully!")

    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": typeof_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_persons,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])

    st.subheader("📊 Input Summary")
    st.json(input_data)

    # 👉 Replace this with your model prediction
    st.info("🔮 Prediction result will appear here")
# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'CreditScore': CreditScore,
    'Geography': Geography,
    'Age': Age,
    'Tenure': Tenure,
    'Balance': Balance,
    'NumOfProducts': NumOfProducts,
    'HasCrCard': 1 if HasCrCard == "Yes" else 0,
    'IsActiveMember': 1 if IsActiveMember == "Yes" else 0,
    'EstimatedSalary': EstimatedSalary
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "churn" if prediction == 1 else "not churn"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
