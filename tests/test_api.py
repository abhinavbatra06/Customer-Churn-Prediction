# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

VALID_CUSTOMER = {
    "Customer_ID": "TEST-001",
    "Age": 37,
    "Gender": "Female",
    "Married": "Yes",
    "Number_of_Dependents": 0,
    "Number_of_Referrals": 2,
    "Offer": None,
    "Phone_Service": "Yes",
    "Multiple_Lines": "No",
    "Avg_Monthly_Long_Distance_Charges": 42.39,
    "Internet_Service": "Yes",
    "Internet_Type": "Cable",
    "Avg_Monthly_GB_Download": 16.0,
    "Online_Security": "No",
    "Online_Backup": "Yes",
    "Device_Protection_Plan": "No",
    "Premium_Tech_Support": "Yes",
    "Streaming_TV": "Yes",
    "Streaming_Movies": "No",
    "Streaming_Music": "No",
    "Unlimited_Data": "Yes",
    "Contract": "One Year",
    "Paperless_Billing": "Yes",
    "Payment_Method": "Credit Card"
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid_customer():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_median_survival_months" in body
    assert isinstance(body["predicted_median_survival_months"], float)
    assert body["predicted_median_survival_months"] > 0

def test_predict_returns_customer_id():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.json()["Customer_ID"] == "TEST-001"

def test_predict_missing_required_field():
    bad_payload = {k: v for k, v in VALID_CUSTOMER.items() if k != "Age"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # Pydantic validation error