from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import CustomerInput, PredictionOutput
from model import predict

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts telecom customer churn using Random Forest.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://localhost:80",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "RandomForestClassifier",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_churn(customer: CustomerInput):
    try:
        result = predict(customer)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/features")
def get_features():
    from model import feature_columns, feature_importances
    return {
        "features": feature_columns,
        "top_importances": {
            feat: round(float(imp), 4)
            for feat, imp in feature_importances.head(10).items()
        }
    }