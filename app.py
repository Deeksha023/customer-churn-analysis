"""
Customer Churn Prediction - FastAPI Web Application Backend
Serves real-time inference using the pre-trained Random Forest model
and provides dataset analytics from the Telco Customer Churn dataset.
"""

import os
import json
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ==============================================================================
# CONFIGURABLE RISK THRESHOLDS
# ==============================================================================
# Easily adjust the probability boundaries for risk tiers
RISK_THRESHOLDS = {
    "low_max": 0.30,       # 0.00 to 0.30 (0% - 30%)   -> Low Risk
    "medium_max": 0.60,    # 0.30 to 0.60 (30% - 60%)  -> Medium Risk
    # > 0.60 (60% - 100%)                              -> High Risk
}

# Paths to existing project artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Initialize FastAPI App
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using existing Random Forest ML model",
    version="1.0.0"
)

# Global model and metadata variables
model = None
feature_columns = None
scaler = None
dataset_stats = {}

# ==============================================================================
# LOAD EXISTING MODEL ARTIFACTS & DATASET STATS
# ==============================================================================
def load_ml_artifacts():
    global model, feature_columns, scaler
    model_path = os.path.join(MODELS_DIR, "random_forest_model.joblib")
    columns_path = os.path.join(MODELS_DIR, "model_columns.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")

    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        raise FileNotFoundError(f"Trained model artifacts missing in {MODELS_DIR}")

    model = joblib.load(model_path)
    feature_columns = joblib.load(columns_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    print(f"[OK] Loaded Random Forest model and {len(feature_columns)} feature columns.")

def calculate_dataset_stats():
    global dataset_stats
    if not os.path.exists(DATA_PATH):
        print(f"[WARN] Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    total_customers = int(len(df))
    churned_customers = int((df['Churn'] == 'Yes').sum())
    retained_customers = int((df['Churn'] == 'No').sum())
    churn_rate = round((churned_customers / total_customers) * 100, 2)

    # 1. Churn by Contract Type
    contract_df = df.groupby(['Contract', 'Churn']).size().unstack(fill_value=0)
    contract_data = {
        "labels": list(contract_df.index),
        "retained": [int(contract_df.loc[idx, 'No']) if 'No' in contract_df else 0 for idx in contract_df.index],
        "churned": [int(contract_df.loc[idx, 'Yes']) if 'Yes' in contract_df else 0 for idx in contract_df.index]
    }

    # 2. Churn by Internet Service
    internet_df = df.groupby(['InternetService', 'Churn']).size().unstack(fill_value=0)
    internet_data = {
        "labels": list(internet_df.index),
        "retained": [int(internet_df.loc[idx, 'No']) if 'No' in internet_df else 0 for idx in internet_df.index],
        "churned": [int(internet_df.loc[idx, 'Yes']) if 'Yes' in internet_df else 0 for idx in internet_df.index]
    }

    # 3. Churn by Tenure Cohort
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ['0-12 Mo', '13-24 Mo', '25-36 Mo', '37-48 Mo', '49-60 Mo', '61-72 Mo']
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)
    tenure_df = df.groupby(['tenure_group', 'Churn'], observed=False).size().unstack(fill_value=0)
    tenure_data = {
        "labels": list(tenure_df.index),
        "retained": [int(tenure_df.loc[idx, 'No']) if 'No' in tenure_df else 0 for idx in tenure_df.index],
        "churned": [int(tenure_df.loc[idx, 'Yes']) if 'Yes' in tenure_df else 0 for idx in tenure_df.index]
    }

    # 4. Churn by Payment Method
    payment_df = df.groupby(['PaymentMethod', 'Churn']).size().unstack(fill_value=0)
    payment_data = {
        "labels": list(payment_df.index),
        "retained": [int(payment_df.loc[idx, 'No']) if 'No' in payment_df else 0 for idx in payment_df.index],
        "churned": [int(payment_df.loc[idx, 'Yes']) if 'Yes' in payment_df else 0 for idx in payment_df.index]
    }

    dataset_stats = {
        "kpis": {
            "total_customers": total_customers,
            "churned_customers": churned_customers,
            "retained_customers": retained_customers,
            "overall_churn_rate": churn_rate
        },
        "charts": {
            "contract": contract_data,
            "internet": internet_data,
            "tenure": tenure_data,
            "payment": payment_data
        }
    }
    print("[OK] Dataset statistics & chart data computed successfully.")

# Load at startup
load_ml_artifacts()
calculate_dataset_stats()

# ==============================================================================
# PYDANTIC INPUT SCHEMA
# ==============================================================================
class CustomerInput(BaseModel):
    gender: str = Field(default="Female", description="Gender: Female or Male")
    SeniorCitizen: int = Field(default=0, ge=0, le=1, description="Senior Citizen: 0 or 1")
    Partner: str = Field(default="No", description="Has Partner: Yes or No")
    Dependents: str = Field(default="No", description="Has Dependents: Yes or No")
    tenure: float = Field(default=12.0, ge=0.0, le=100.0, description="Tenure in months")
    PhoneService: str = Field(default="Yes", description="Phone Service: Yes or No")
    MultipleLines: str = Field(default="No", description="Multiple Lines: No, Yes, or No phone service")
    InternetService: str = Field(default="Fiber optic", description="Internet Service: DSL, Fiber optic, No")
    OnlineSecurity: str = Field(default="No", description="Online Security: Yes, No, No internet service")
    OnlineBackup: str = Field(default="No", description="Online Backup: Yes, No, No internet service")
    DeviceProtection: str = Field(default="No", description="Device Protection: Yes, No, No internet service")
    TechSupport: str = Field(default="No", description="Tech Support: Yes, No, No internet service")
    StreamingTV: str = Field(default="No", description="Streaming TV: Yes, No, No internet service")
    StreamingMovies: str = Field(default="No", description="Streaming Movies: Yes, No, No internet service")
    Contract: str = Field(default="Month-to-month", description="Contract: Month-to-month, One year, Two year")
    PaperlessBilling: str = Field(default="Yes", description="Paperless Billing: Yes or No")
    PaymentMethod: str = Field(
        default="Electronic check", 
        description="Payment Method: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)"
    )
    MonthlyCharges: float = Field(default=75.0, ge=10.0, le=150.0, description="Monthly Charges in USD")
    TotalCharges: Optional[float] = Field(default=None, ge=0.0, description="Total Charges in USD (auto-computed if null)")

# ==============================================================================
# FEATURE ENCODER (MATCHES EXISTING TRAINED MODEL COLUMNS EXACTLY)
# ==============================================================================
def encode_customer_profile(data: CustomerInput) -> pd.DataFrame:
    """
    Transforms raw customer inputs into the exact 30 one-hot encoded columns
    saved in models/model_columns.joblib.
    """
    # Create single-row DataFrame initialized to 0 for all 30 expected columns
    row = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # 1. Numerical & direct binary features
    row['SeniorCitizen'] = int(data.SeniorCitizen)
    row['tenure'] = float(data.tenure)
    row['MonthlyCharges'] = float(data.MonthlyCharges)
    
    # Calculate TotalCharges if not provided
    if data.TotalCharges is not None and data.TotalCharges > 0:
        row['TotalCharges'] = float(data.TotalCharges)
    else:
        row['TotalCharges'] = float(max(1.0, data.tenure) * data.MonthlyCharges)

    # 2. Categorical one-hot variables
    if data.gender == "Male":
        row['gender_Male'] = 1
    if data.Partner == "Yes":
        row['Partner_Yes'] = 1
    if data.Dependents == "Yes":
        row['Dependents_Yes'] = 1
    if data.PhoneService == "Yes":
        row['PhoneService_Yes'] = 1

    if data.MultipleLines == "No phone service":
        row['MultipleLines_No phone service'] = 1
    elif data.MultipleLines == "Yes":
        row['MultipleLines_Yes'] = 1

    if data.InternetService == "Fiber optic":
        row['InternetService_Fiber optic'] = 1
    elif data.InternetService == "No":
        row['InternetService_No'] = 1

    # Internet sub-services
    services = [
        ('OnlineSecurity', data.OnlineSecurity),
        ('OnlineBackup', data.OnlineBackup),
        ('DeviceProtection', data.DeviceProtection),
        ('TechSupport', data.TechSupport),
        ('StreamingTV', data.StreamingTV),
        ('StreamingMovies', data.StreamingMovies),
    ]
    for service_name, val in services:
        if val == "No internet service":
            col_no_int = f"{service_name}_No internet service"
            if col_no_int in feature_columns:
                row[col_no_int] = 1
        elif val == "Yes":
            col_yes = f"{service_name}_Yes"
            if col_yes in feature_columns:
                row[col_yes] = 1

    # Contract
    if data.Contract == "One year":
        row['Contract_One year'] = 1
    elif data.Contract == "Two year":
        row['Contract_Two year'] = 1

    # Paperless Billing
    if data.PaperlessBilling == "Yes":
        row['PaperlessBilling_Yes'] = 1

    # Payment Method
    pm = data.PaymentMethod
    if pm in ["Credit card (automatic)", "Electronic check", "Mailed check"]:
        col_pm = f"PaymentMethod_{pm}"
        if col_pm in feature_columns:
            row[col_pm] = 1

    return row

def generate_risk_explanation(data: CustomerInput, prob: float, risk_level: str) -> List[str]:
    """Generates evidence-based explanations strictly tied to customer values and dataset findings."""
    factors = []
    
    if risk_level in ["High", "Medium"]:
        if data.Contract == "Month-to-month":
            factors.append("Contract: Month-to-month subscription (highest historical churn rate of 42.7%).")
        if data.tenure <= 12:
            factors.append(f"Tenure: Short relationship ({int(data.tenure)} months); customers in their first year experience the highest attrition (47.4%).")
        if data.MonthlyCharges >= 70.0:
            factors.append(f"Monthly Cost: High monthly charge of ${data.MonthlyCharges:.2f}, creating pricing sensitivity.")
        if data.InternetService == "Fiber optic":
            factors.append("Internet Service: Fiber optic service (frequently linked with higher churn if not paired with technical support).")
        if data.TechSupport in ["No", "No internet service"]:
            factors.append("Support: Lacks dedicated Tech Support add-on.")
        if data.OnlineSecurity in ["No", "No internet service"]:
            factors.append("Security: Lacks Online Security package.")
        if data.PaymentMethod == "Electronic check":
            factors.append("Payment Method: Electronic check payments have the highest historical churn rate (45.3%).")
        if data.PaperlessBilling == "Yes" and data.Contract == "Month-to-month":
            factors.append("Billing: Paperless billing without automatic card or bank payment.")
    else:
        # Low risk factors
        if data.Contract in ["One year", "Two year"]:
            factors.append(f"Contract: Long-term {data.Contract.lower()} commitment drastically lowers cancellation risk.")
        if data.tenure >= 24:
            factors.append(f"Tenure: Established tenure of {int(data.tenure)} months indicates strong customer loyalty.")
        if data.TechSupport == "Yes":
            factors.append("Support: Subscribed to dedicated Tech Support, enhancing service stickiness.")
        if data.OnlineSecurity == "Yes":
            factors.append("Security: Active Online Security package increases switching barriers.")
        if data.PaymentMethod in ["Credit card (automatic)", "Bank transfer (automatic)"]:
            factors.append(f"Payment Method: Automatic payment ({data.PaymentMethod}) stabilizes recurring billing.")
        if data.MonthlyCharges < 60.0:
            factors.append(f"Monthly Cost: Affordable monthly charge of ${data.MonthlyCharges:.2f}.")

    if not factors:
        factors.append("Balanced customer profile across contract, tenure, and service options.")
        
    return factors

def generate_recommendations(risk_level: str) -> List[str]:
    """Provides actionable business recommendations separated from ML model outputs."""
    if risk_level == "High":
        return [
            "Contract Incentive: Offer a 15-20% discount on migrating to an annual (1-year or 2-year) contract commitment.",
            "Service Bundle: Provide a complimentary 3-month trial of Premium Tech Support and Online Security.",
            "Proactive Outreach: Assign an onboarding specialist to verify satisfaction and resolve connection friction.",
            "Payment Optimization: Offer a $5 bill credit for setting up automatic bank transfer or credit card payments."
        ]
    elif risk_level == "Medium":
        return [
            "Customer Engagement: Send a mid-term pulse check survey to capture satisfaction and service feedback.",
            "Value Promotion: Highlight bundled package savings for adding streaming or security services.",
            "Contract Upgrade: Present an annual renewal incentive prior to upcoming billing cycles.",
            "Billing Review: Review monthly usage to ensure the plan fits the customer's current consumption."
        ]
    else:
        return [
            "Loyalty Reward: Enroll the customer in an exclusive VIP loyalty program to reward continued tenure.",
            "Service Continuity: Maintain consistent service quality and quarterly automated check-ins.",
            "Referral Program: Encourage word-of-mouth growth by offering a referral credit for introducing new subscribers.",
            "Product Upgrades: Offer early access to upcoming high-speed bandwidth or device upgrade options."
        ]

# ==============================================================================
# API ENDPOINTS
# ==============================================================================
@app.get("/api/stats")
def get_stats():
    """Returns dataset summary statistics and chart distributions."""
    return dataset_stats

@app.get("/api/thresholds")
def get_thresholds():
    """Returns the current risk tier thresholds."""
    return RISK_THRESHOLDS

@app.post("/api/predict")
def predict_churn_api(customer: CustomerInput):
    """
    Receives customer profile, encodes into model features, runs Random Forest
    inference, and returns churn prediction, probability, risk level, and recommendations.
    """
    if model is None or feature_columns is None:
        raise HTTPException(status_code=500, detail="ML model artifacts not loaded")

    # Encode input into DataFrame matching 30 feature columns
    encoded_df = encode_customer_profile(customer)

    # Perform inference with existing trained Random Forest
    raw_pred = int(model.predict(encoded_df)[0])
    raw_prob = float(model.predict_proba(encoded_df)[0][1])
    prob_percent = round(raw_prob * 100, 2)

    # Classify risk level based on configurable thresholds
    if raw_prob < RISK_THRESHOLDS["low_max"]:
        risk_level = "Low"
        prediction_label = "LOW RISK - CUSTOMER LIKELY RETAINED"
    elif raw_prob < RISK_THRESHOLDS["medium_max"]:
        risk_level = "Medium"
        prediction_label = "MEDIUM RISK - ATTENTION RECOMMENDED"
    else:
        risk_level = "High"
        prediction_label = "HIGH RISK - CUSTOMER MAY CHURN"

    # Explanations and business recommendations
    explanations = generate_risk_explanation(customer, raw_prob, risk_level)
    recommendations = generate_recommendations(risk_level)

    return {
        "prediction_label": prediction_label,
        "churn_binary": raw_pred,
        "churn_probability_percent": prob_percent,
        "risk_level": risk_level,
        "risk_thresholds": {
            "low": f"0% - {int(RISK_THRESHOLDS['low_max']*100)}%",
            "medium": f"{int(RISK_THRESHOLDS['low_max']*100)}% - {int(RISK_THRESHOLDS['medium_max']*100)}%",
            "high": f"{int(RISK_THRESHOLDS['medium_max']*100)}% - 100%"
        },
        "explanations": explanations,
        "recommendations": recommendations,
        "model_used": "Random Forest Classifier (100 Trees, models/random_forest_model.joblib)"
    }

# Mount static folder
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_frontend():
    """Serves the main web dashboard."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Customer Churn Prediction API is running. Static frontend not yet compiled."}

if __name__ == "__main__":
    import uvicorn
    print("\nStarting Customer Churn Web Application on http://127.0.0.1:8000...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
