# api/main.py

import pandas as pd
from fastapi import FastAPI, HTTPException
from api.schemas import CustomerInput, PredictionResponse
from src.predict import predict_survival

app = FastAPI(title="Churn Survival Prediction API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput): # customer is a instance of CustomerInput, which is a Pydantic model
    df = pd.DataFrame([customer.model_dump()])
    try:
        result = predict_survival(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(
        Customer_ID=result["Customer_ID"].iloc[0] if "Customer_ID" in result.columns else None,
        predicted_median_survival_months=round(float(result["predicted_median_survival_months"].iloc[0]), 2)
    )