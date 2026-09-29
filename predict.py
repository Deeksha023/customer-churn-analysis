"""
Customer Churn Prediction - Standalone Inference Script
This script demonstrates loading the trained model and making real-time predictions
for new customer profiles.
"""

import os
import joblib
import pandas as pd
import numpy as np

def load_artifacts(models_dir="models"):
    """Load saved model and column metadata."""
    model_path = os.path.join(models_dir, "random_forest_model.joblib")
    columns_path = os.path.join(models_dir, "model_columns.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        raise FileNotFoundError(
            f"Model artifacts not found in '{models_dir}'. Please run the training notebook first."
        )
    
    model = joblib.load(model_path)
    columns = joblib.load(columns_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    return model, columns, scaler

def predict_churn(customer_data: dict, model, feature_columns) -> dict:
    """
    Predict churn probability and binary label for a customer dictionary.
    
    Args:
        customer_data: dict with customer attributes (e.g. tenure, MonthlyCharges, Contract, etc.)
        model: loaded scikit-learn model
        feature_columns: list of feature column names expected by the model
        
    Returns:
        dict with prediction results: label, probability, risk_level
    """
    # Create an empty template with all expected encoded columns initialized to 0
    row_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Populate continuous numerical features
    if 'tenure' in customer_data:
        row_df['tenure'] = customer_data['tenure']
    if 'MonthlyCharges' in customer_data:
        row_df['MonthlyCharges'] = customer_data['MonthlyCharges']
    if 'TotalCharges' in customer_data:
        row_df['TotalCharges'] = customer_data['TotalCharges']
    else:
        # Approximate TotalCharges if not provided
        row_df['TotalCharges'] = customer_data.get('tenure', 1) * customer_data.get('MonthlyCharges', 50.0)
        
    if 'SeniorCitizen' in customer_data:
        row_df['SeniorCitizen'] = customer_data['SeniorCitizen']

    # Populate categorical one-hot flags based on dictionary inputs
    for key, value in customer_data.items():
        # Check direct column matches (e.g. Contract_One year, Partner_Yes)
        col_name = f"{key}_{value}"
        if col_name in feature_columns:
            row_df[col_name] = 1

    # Predict
    pred = int(model.predict(row_df)[0])
    prob = float(model.predict_proba(row_df)[0][1])
    
    risk_level = "High" if prob >= 0.65 else ("Moderate" if prob >= 0.40 else "Low")
    
    return {
        "churn_prediction": "Yes (Churn)" if pred == 1 else "No (Retained)",
        "churn_probability_percent": round(prob * 100, 2),
        "risk_level": risk_level
    }

def main():
    print("=" * 65)
    print("      CUSTOMER CHURN PREDICTION - REAL-TIME INFERENCE DEMO      ")
    print("=" * 65)
    
    # 1. Load artifacts
    print("\n[1] Loading trained model artifacts from 'models/'...")
    model, columns, _ = load_artifacts()
    print("    -> Successfully loaded Random Forest Classifier.")
    print(f"    -> Total input features expected: {len(columns)}")

    # 2. Test Customer A: High-Risk Profile
    # Month-to-month, new customer (2 months), high monthly charges ($95), Fiber optic, Electronic check
    customer_a = {
        "tenure": 2,
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0,
        "SeniorCitizen": 0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes"
    }

    # 3. Test Customer B: Low-Risk Profile
    # Two-year contract, 60 months tenure, moderate charges ($45), Tech Support, Online Security
    customer_b = {
        "tenure": 60,
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0,
        "SeniorCitizen": 0,
        "Contract": "Two year",
        "InternetService": "DSL",
        "TechSupport": "Yes",
        "OnlineSecurity": "Yes",
        "Partner": "Yes",
        "Dependents": "Yes"
    }

    print("\n[2] Evaluating Sample Customer Profiles:")
    print("-" * 65)
    
    result_a = predict_churn(customer_a, model, columns)
    print("PROFILE A: New Customer on Month-to-Month Plan ($95.00/mo, Fiber Optic)")
    print(f"  -> Prediction:        {result_a['churn_prediction']}")
    print(f"  -> Churn Probability: {result_a['churn_probability_percent']}%")
    print(f"  -> Risk Level:        {result_a['risk_level']}")
    print("-" * 65)

    result_b = predict_churn(customer_b, model, columns)
    print("PROFILE B: Loyal Customer on 2-Year Contract ($45.00/mo, Tech Support)")
    print(f"  -> Prediction:        {result_b['churn_prediction']}")
    print(f"  -> Churn Probability: {result_b['churn_probability_percent']}%")
    print(f"  -> Risk Level:        {result_b['risk_level']}")
    print("-" * 65)
    
    print("\nInference successfully demonstrated!")

if __name__ == "__main__":
    main()
