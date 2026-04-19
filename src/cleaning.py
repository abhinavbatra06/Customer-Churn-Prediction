


import pandas as pd
import numpy as np

def clean_churn_data(df: pd.DataFrame):
    """
    Clean raw telecom churn data.
    
    Args:
        df: Raw dataframe from telecom_customer_churn.csv
    
    Returns:
        Cleaned dataframe ready for feature engineering
    """
    df = df.copy()
    
    # Standardize column names
    df.columns = df.columns.str.replace(' ', '_')
    
    # --- Internet features ---
    internet_binary_features = [
        'Online_Security', 'Online_Backup', 'Device_Protection_Plan',
        'Premium_Tech_Support', 'Streaming_TV', 'Streaming_Movies',
        'Streaming_Music', 'Unlimited_Data'
    ]
    
    df[internet_binary_features] = df[internet_binary_features].replace({'Yes': 1, 'No': 0})
    df.loc[df['Internet_Service'] == 'No', internet_binary_features] = 0
    df['Num_Internet_Features'] = df[internet_binary_features].sum(axis=1) + 1
    df.loc[df['Internet_Service'] == 'No', 'Num_Internet_Features'] = 0
    df.loc[df['Internet_Service'] == 'No', 'Internet_Type'] = 'None'
    df.loc[df['Internet_Service'] == 'No', 'Avg_Monthly_GB_Download'] = 0
    df.drop(columns=['Internet_Service'] + internet_binary_features, inplace=True)
    
    # --- Phone features ---
    df.loc[df['Phone_Service'] == 'No', 'Avg_Monthly_Long_Distance_Charges'] = 0
    df['Multiple_Lines'] = df['Multiple_Lines'].replace({'Yes': 1, 'No': 0})
    df.loc[df['Phone_Service'] == 'No', 'Multiple_Lines'] = 0
    df['Has_Multiple_Lines'] = df['Multiple_Lines'] + 1
    df.loc[df['Phone_Service'] == 'No', 'Has_Multiple_Lines'] = 0
    df.drop(columns=['Multiple_Lines', 'Phone_Service'], inplace=True)
    
    # --- Offer ---
    df['Offer'] = df['Offer'].fillna('No Offer')
    
    # --- Churn status --- ( Only add it for training data, not for prediction data )
    if 'Customer_Status' in df.columns:
        condition = [
            df['Customer_Status'].isin(['Stayed', 'Joined']),
            df['Customer_Status'].isin(['Churned'])
        ]
        df['Churn_status'] = np.select(condition, [0, 1], default=0)
    
    return df