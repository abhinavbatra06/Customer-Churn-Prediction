# tests/test_feature_engineering.py

import pandas as pd
import pytest
from src.cleaning import clean_churn_data
from src.feature_engineering import engineer_features

@pytest.fixture
def cleaned_df():
    raw = pd.DataFrame([
        {
            "Customer ID": "0002-ORFBO",
            "Gender": "Female", "Age": 37, "Married": "Yes",
            "Number of Dependents": 0, "City": "Frazier Park",
            "Zip Code": 93225, "Latitude": 34.83, "Longitude": -119.0,
            "Number of Referrals": 2, "Tenure in Months": 9,
            "Offer": "Offer A",
            "Phone Service": "Yes", "Multiple Lines": "No",
            "Internet Service": "Yes", "Internet Type": "Cable",
            "Avg Monthly Long Distance Charges": 42.39,
            "Avg Monthly GB Download": 16,
            "Online Security": "No", "Online Backup": "Yes",
            "Device Protection Plan": "No", "Premium Tech Support": "Yes",
            "Streaming TV": "Yes", "Streaming Movies": "No",
            "Streaming Music": "No", "Unlimited Data": "Yes",
            "Contract": "One Year", "Paperless Billing": "Yes",
            "Payment Method": "Credit Card",
            "Monthly Charge": 65.6, "Total Charges": 593.3,
            "Total Refunds": 0, "Total Extra Data Charges": 0,
            "Total Long Distance Charges": 381.51, "Total Revenue": 974.81,
            "Customer Status": "Stayed", "Churn Category": None, "Churn Reason": None
        },
        {
            "Customer ID": "TEST-002",
            "Gender": "Male", "Age": 45, "Married": "No",
            "Number of Dependents": 0, "City": "LA",
            "Zip Code": 90001, "Latitude": 34.0, "Longitude": -118.0,
            "Number of Referrals": 0, "Tenure in Months": 24,
            "Offer": None,
            "Phone Service": "Yes", "Multiple Lines": "Yes",
            "Internet Service": "Yes", "Internet Type": "Fiber Optic",
            "Avg Monthly Long Distance Charges": 10.0,
            "Avg Monthly GB Download": 50,
            "Online Security": "Yes", "Online Backup": "No",
            "Device Protection Plan": "Yes", "Premium Tech Support": "No",
            "Streaming TV": "No", "Streaming Movies": "Yes",
            "Streaming Music": "Yes", "Unlimited Data": "No",
            "Contract": "Month-to-Month", "Paperless Billing": "No",
            "Payment Method": "Bank Withdrawal",
            "Monthly Charge": 90.0, "Total Charges": 2160.0,
            "Total Refunds": 0, "Total Extra Data Charges": 0,
            "Total Long Distance Charges": 240.0, "Total Revenue": 2400.0,
            "Customer Status": "Churned", "Churn Category": None, "Churn Reason": None
        }
    ])
    return clean_churn_data(raw)

def test_geography_columns_dropped(cleaned_df):
    result = engineer_features(cleaned_df)
    for col in ['City', 'Zip_Code', 'Latitude', 'Longitude']:
        assert col not in result.columns

def test_gender_encoded(cleaned_df):
    result = engineer_features(cleaned_df)
    assert 'gender_e' in result.columns
    assert result['gender_e'].iloc[0] == 0  

def test_married_encoded(cleaned_df):
    result = engineer_features(cleaned_df)
    assert result['married_e'].iloc[0] == 1  

def test_offer_encoded(cleaned_df):
    result = engineer_features(cleaned_df)
    assert result['offer_e'].iloc[0] == 1  

def test_contract_encoded(cleaned_df):
    result = engineer_features(cleaned_df)
    assert result['Contract_e'].iloc[0] == 1  

def test_payment_method_encoded(cleaned_df):
    result = engineer_features(cleaned_df)
    assert result['Payment_Method_e'].iloc[0] == 1  

def test_internet_type_one_hot(cleaned_df):
    result = engineer_features(cleaned_df)
    # Cable should produce Internet_Type_Fiber Optic column (drop_first drops Cable)
    assert any(col.startswith('Internet_Type_') for col in result.columns)

def test_original_categorical_cols_dropped(cleaned_df):
    result = engineer_features(cleaned_df)
    for col in ['Gender', 'Married', 'Paperless_Billing', 'Offer', 'Contract', 'Payment_Method']:
        assert col not in result.columns