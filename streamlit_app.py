import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Churn Survival Predictor", page_icon="📡", layout="centered")

st.title("📡 Customer Churn Survival Predictor")
st.markdown(
    "Predict **how long a telecom customer will stay** before churning, "
    "using a Log-Normal AFT survival model."
)
st.divider()

# --- Customer Identity ---
st.subheader("Customer")
col1, col2, col3 = st.columns(3)
customer_id = col1.text_input("Customer ID (optional)")
age         = col2.number_input("Age", min_value=18, max_value=100, value=35)
gender      = col3.selectbox("Gender", ["Male", "Female"])

col4, col5 = st.columns(2)
married              = col4.selectbox("Married", ["Yes", "No"])
num_dependents       = col5.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
num_referrals        = st.number_input("Number of Referrals", min_value=0, max_value=20, value=0)

st.divider()

# --- Contract & Billing ---
st.subheader("Contract & Billing")
col6, col7, col8 = st.columns(3)
contract           = col6.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
paperless_billing  = col7.selectbox("Paperless Billing", ["Yes", "No"])
payment_method     = col8.selectbox("Payment Method", ["Credit Card", "Bank Withdrawal", "Mailed Check"])
offer              = st.selectbox("Offer", ["No Offer", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])

st.divider()

# --- Phone Service ---
st.subheader("Phone Service")
col9, col10, col11 = st.columns(3)
phone_service      = col9.selectbox("Phone Service", ["Yes", "No"])
multiple_lines     = col10.selectbox("Multiple Lines", ["Yes", "No"])
avg_ld_charges     = col11.number_input("Avg Monthly Long Distance Charges ($)", min_value=0.0, value=20.0)

st.divider()

# --- Internet Service ---
st.subheader("Internet Service")
col12, col13, col14 = st.columns(3)
internet_service   = col12.selectbox("Internet Service", ["Yes", "No"])
internet_type      = col13.selectbox("Internet Type", ["Fiber Optic", "Cable", "DSL", "None"])
avg_gb_download    = col14.number_input("Avg Monthly GB Download", min_value=0.0, value=50.0)

st.markdown("**Internet Add-ons**")
col_a, col_b, col_c, col_d = st.columns(4)
online_security        = col_a.selectbox("Online Security",        ["Yes", "No"])
online_backup          = col_b.selectbox("Online Backup",          ["Yes", "No"])
device_protection      = col_c.selectbox("Device Protection Plan", ["Yes", "No"])
premium_tech_support   = col_d.selectbox("Premium Tech Support",   ["Yes", "No"])

col_e, col_f, col_g, col_h = st.columns(4)
streaming_tv    = col_e.selectbox("Streaming TV",     ["Yes", "No"])
streaming_movies= col_f.selectbox("Streaming Movies", ["Yes", "No"])
streaming_music = col_g.selectbox("Streaming Music",  ["Yes", "No"])
unlimited_data  = col_h.selectbox("Unlimited Data",   ["Yes", "No"])

st.divider()

# --- Predict ---
if st.button("Predict Survival Time", type="primary", use_container_width=True):
    payload = {
        "Customer_ID":                        customer_id or None,
        "Age":                                age,
        "Gender":                             gender,
        "Married":                            married,
        "Number_of_Dependents":               num_dependents,
        "Number_of_Referrals":                num_referrals,
        "Offer":                              offer,
        "Phone_Service":                      phone_service,
        "Multiple_Lines":                     multiple_lines,
        "Avg_Monthly_Long_Distance_Charges":  avg_ld_charges,
        "Internet_Service":                   internet_service,
        "Internet_Type":                      internet_type,
        "Avg_Monthly_GB_Download":            avg_gb_download,
        "Online_Security":                    online_security,
        "Online_Backup":                      online_backup,
        "Device_Protection_Plan":             device_protection,
        "Premium_Tech_Support":               premium_tech_support,
        "Streaming_TV":                       streaming_tv,
        "Streaming_Movies":                   streaming_movies,
        "Streaming_Music":                    streaming_music,
        "Unlimited_Data":                     unlimited_data,
        "Contract":                           contract,
        "Paperless_Billing":                  paperless_billing,
        "Payment_Method":                     payment_method,
    }

    with st.spinner("Predicting..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            months = result["predicted_median_survival_months"]
            st.success("Prediction complete")
            st.metric(
                label="Predicted Median Tenure",
                value=f"{months} months",
                help="The model predicts this customer will stay for this many months before churning."
            )

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure the backend is running.")
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e.response.status_code} — {e.response.text}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
