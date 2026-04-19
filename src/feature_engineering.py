# src/feature_engineering.py

import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode and transform cleaned data for modeling.
    
    Args:
        df: Cleaned dataframe from cleaning step
    
    Returns:
        Encoded dataframe ready for AFT/Cox models
    """
    df = df.copy()
    
    # --- Remove columns not needed for modeling ---
    to_remove = [
        'Customer_ID', 'City', 'Zip_Code', 'Latitude', 'Longitude',
        'Total_Charges', 'Total_Refunds', 'Total_Extra_Data_Charges',
        'Total_Long_Distance_Charges', 'Churn_Category', 'Churn_Reason',
        'Customer_Status', 'Total_Revenue', 'Monthly_Charge'
    ]
    df = df.drop(columns=[c for c in to_remove if c in df.columns])
    
    # --- Binary encoding (2-category columns) ---
    df['gender_e'] = (df['Gender'] == 'Male').astype(int)
    df['married_e'] = (df['Married'] == 'Yes').astype(int)
    df['paper_e'] = (df['Paperless_Billing'] == 'Yes').astype(int)
    
    # --- Custom encoding: Offer (any offer → 1) ---
    df['offer_e'] = df['Offer'].isin(
        ['Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E']
    ).astype(int)
    
    # --- Custom encoding: Contract (1-2 year → 1) ---
    df['Contract_e'] = df['Contract'].isin(['Two Year', 'One Year']).astype(int)
    
    # --- Custom encoding: Payment Method (Credit Card → 1) ---
    df['Payment_Method_e'] = (df['Payment_Method'] == 'Credit Card').astype(int)
    
    # --- One-hot encode Internet_Type ---
    df['Internet_Type_Fiber Optic'] = (df['Internet_Type'] == 'Fiber Optic').astype(int)
    df['Internet_Type_None'] = (df['Internet_Type'] == 'None').astype(int)
    df = df.drop(columns=['Internet_Type'])
    
    # --- Drop original categorical columns ---
    cols_to_drop = ['Gender', 'Married', 'Paperless_Billing', 'Offer', 'Contract', 'Payment_Method']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df