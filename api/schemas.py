# api/schemas.py

from pydantic import BaseModel
from typing import Optional

class CustomerInput(BaseModel):
    Customer_ID: Optional[str] = None
    Age: int
    Gender: str                             # Male / Female
    Married: str                            # Yes / No
    Number_of_Dependents: int
    Number_of_Referrals: int
    Offer: Optional[str] = None             # Offer A-E or None
    Phone_Service: str                      # Yes / No
    Multiple_Lines: str                     # Yes / No
    Avg_Monthly_Long_Distance_Charges: float
    Internet_Service: str                   # Yes / No
    Internet_Type: Optional[str] = None     # Cable / Fiber Optic / DSL / None
    Avg_Monthly_GB_Download: float
    Online_Security: str                    # Yes / No
    Online_Backup: str                      # Yes / No
    Device_Protection_Plan: str             # Yes / No
    Premium_Tech_Support: str               # Yes / No
    Streaming_TV: str                       # Yes / No
    Streaming_Movies: str                   # Yes / No
    Streaming_Music: str                    # Yes / No
    Unlimited_Data: str                     # Yes / No
    Contract: str                           # Month-to-Month / One Year / Two Year
    Paperless_Billing: str                  # Yes / No
    Payment_Method: str                     # Credit Card / Bank Withdrawal / Mailed Check

class PredictionResponse(BaseModel):
    Customer_ID: Optional[str] = None
    predicted_median_survival_months: float