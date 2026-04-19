# tests/test_cleaning.py

import pandas as pd
import numpy as np
import pytest
from src.cleaning import clean_churn_data

@pytest.fixture
def raw_row():
    # sample raw data 
    return pd.DataFrame([{
        "Customer ID": "0002-ORFBO",
        "Gender": "Female", "Age": 37, "Married": "Yes",
        "Number of Dependents": 0, "City": "Frazier Park",
        "Zip Code": 93225, "Latitude": 34.83, "Longitude": -119.0,
        "Number of Referrals": 2, "Tenure in Months": 9,
        "Offer": None,
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
    }])

def test_column_names_standardized(raw_row):
    ''' check if column names are standardized (lowercase, underscores) '''
    result = clean_churn_data(raw_row)
    assert all(' ' not in col for col in result.columns)

def test_churn_status_created(raw_row):
    ''' check if churn_status column is created correctly but only during training, not during prediction  '''

    result = clean_churn_data(raw_row)
    assert 'Churn_status' in result.columns
    assert result['Churn_status'].iloc[0] == 0  

def test_internet_service_dropped(raw_row):
    ''' check if Internet_Service column is dropped '''
    result = clean_churn_data(raw_row)
    assert 'Internet_Service' not in result.columns

def test_phone_service_dropped(raw_row):
    ''' check if Phone_Service column is dropped '''

    result = clean_churn_data(raw_row)
    assert 'Phone_Service' not in result.columns

def test_null_offer_filled(raw_row):    
    ''' check if null Offer values are filled with 'No Offer' '''
    result = clean_churn_data(raw_row)
    assert result['Offer'].iloc[0] == 'No Offer'

def test_no_internet_customer():
    ''' check if Num_Internet_Features is 0 for customers with no internet service '''
    df = pd.DataFrame([{
        "Customer ID": "TEST-001",
        "Gender": "Male", "Age": 40, "Married": "No",
        "Number of Dependents": 0, "City": "LA",
        "Zip Code": 90001, "Latitude": 34.0, "Longitude": -118.0,
        "Number of Referrals": 0, "Tenure in Months": 12,
        "Offer": None,
        "Phone Service": "Yes", "Multiple Lines": "No",
        "Internet Service": "No", "Internet Type": None,
        "Avg Monthly Long Distance Charges": 10.0,
        "Avg Monthly GB Download": 0,
        "Online Security": "No", "Online Backup": "No",
        "Device Protection Plan": "No", "Premium Tech Support": "No",
        "Streaming TV": "No", "Streaming Movies": "No",
        "Streaming Music": "No", "Unlimited Data": "No",
        "Contract": "Month-to-Month", "Paperless Billing": "No",
        "Payment Method": "Bank Withdrawal",
        "Monthly Charge": 20.0, "Total Charges": 240.0,
        "Total Refunds": 0, "Total Extra Data Charges": 0,
        "Total Long Distance Charges": 120.0, "Total Revenue": 360.0,
        "Customer Status": "Churned", "Churn Category": None, "Churn Reason": None
    }])
    result = clean_churn_data(df)
    assert result['Num_Internet_Features'].iloc[0] == 0
    assert result['Churn_status'].iloc[0] == 1