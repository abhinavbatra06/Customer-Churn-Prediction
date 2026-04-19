# src/predict.py

import pandas as pd
import joblib
from src.cleaning import clean_churn_data
from src.feature_engineering import engineer_features
from src.config import Paths, ModelConfig

def load_model():
    return joblib.load(Paths.model)

# columns to drop before prediction (same as during training)
cols_to_drop = ModelConfig.cols_to_drop + ['Churn_status', 'Tenure_in_Months']


def predict_survival(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict median survival time for customers.
    
    Args:
        df: Raw customer data (same format as original CSV)
    
    Returns:
        DataFrame with Customer_ID and predicted median survival months
    """
    # Store IDs before processing
    customer_ids = df['Customer_ID'].copy() if 'Customer_ID' in df.columns else None
    
    # Process through pipeline
    df_clean = clean_churn_data(df)
    df_encoded = engineer_features(df_clean)

    # Drop target columns and skewed features not used during training
    df_encoded = df_encoded.drop(columns=[c for c in cols_to_drop if c in df_encoded.columns])
    
    # Load model and predict
    model = load_model()
    predictions = model.predict_median(df_encoded)
    
    result = pd.DataFrame({
        'predicted_median_survival_months': predictions
    })
    
    if customer_ids is not None:
        result.insert(0, 'Customer_ID', customer_ids.values)
    
    return result