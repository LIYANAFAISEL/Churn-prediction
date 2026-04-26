import joblib
import pandas as pd
import numpy as np
import os
from schemas import CustomerInput, PredictionOutput, FeatureFactor

# Paths — go one level up from api/ to find pkl files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH   = os.path.join(BASE_DIR, 'churn_model.pkl')
SCALER_PATH  = os.path.join(BASE_DIR, 'scaler.pkl')
COLUMNS_PATH = os.path.join(BASE_DIR, 'feature_columns.pkl')

# Load once at startup
print("Loading model...")
model           = joblib.load(MODEL_PATH)
scaler          = joblib.load(SCALER_PATH)
feature_columns = joblib.load(COLUMNS_PATH)

feature_importances = pd.Series(
    model.feature_importances_,
    index=feature_columns
).sort_values(ascending=False)

print(f"Model loaded: {type(model).__name__}")
print(f"Features expected: {len(feature_columns)}")


def preprocess(customer: CustomerInput) -> pd.DataFrame:
    data = customer.dict()
    df = pd.DataFrame([data])

    # Encode gender
    df['gender'] = (df['gender'] == 'Male').astype(int)

    # Encode Yes/No columns
    yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in yes_no_cols:
        df[col] = (df[col] == 'Yes').astype(int)

    # Encode three-value columns
    three_val_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                      'DeviceProtection', 'TechSupport',
                      'StreamingTV', 'StreamingMovies']
    for col in three_val_cols:
        df[col] = df[col].replace({
            'No phone service':    'No',
            'No internet service': 'No'
        })
        df[col] = (df[col] == 'Yes').astype(int)

    # One-hot encode
    df = pd.get_dummies(
        df,
        columns=['InternetService', 'Contract', 'PaymentMethod']
    )

    # Align columns — add missing ones as 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df


def predict(customer: CustomerInput) -> PredictionOutput:
    df = preprocess(customer)

    churn_prob = float(model.predict_proba(df)[0][1])

    if churn_prob >= 0.70:
        risk_level = 'High'
    elif churn_prob >= 0.40:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'

    prediction = 'Will Churn' if churn_prob >= 0.5 else 'Will Stay'

    top_factors = [
        FeatureFactor(
            feature=feat,
            importance=round(float(imp), 4),
            importance_pct=f"{imp*100:.1f}%"
        )
        for feat, imp in feature_importances.head(5).items()
    ]

    return PredictionOutput(
        churn_probability=round(churn_prob, 4),
        churn_probability_pct=f"{churn_prob*100:.1f}%",
        risk_level=risk_level,
        prediction=prediction,
        top_factors=top_factors
    )