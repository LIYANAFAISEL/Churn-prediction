# Customer Churn Prediction

Predicting customer churn for a telecom company using the Telco Customer Churn dataset.

## Tech stack
| Layer | Technology |
|---|---|
| Machine learning | Python, scikit-learn (Random Forest) |
| Backend API | FastAPI |
| Frontend | Angular |
| Containerisation | Docker |

## Project phases
- [x] Phase 1: Data exploration and EDA
- [x] Phase 2: Preprocessing, model training and feature importance
- [x] Phase 3: FastAPI backend — 4 endpoints, tested and working
- [ ] Phase 4: Angular frontend
- [ ] Phase 5: Docker + deployment

## Phase 1 — Key findings from EDA
- 26.5% overall churn rate — roughly 1 in 4 customers leave
- Electronic check payment has the highest churn rate at 45.3%
- Month-to-month contracts churn at 42.7% vs 2.8% for two-year contracts
- Most churn happens within the first 12 months of tenure
- Fiber optic customers churn at 41.9% vs DSL at 19.0%
- Higher monthly charges correlate with higher churn risk
- Gender shows no meaningful difference (26.9% vs 26.2%)
- No internet service customers are lowest risk at 7.4%
- TotalCharges was stored as text — fixed in Phase 2

## Phase 2 — Preprocessing & model results
- Fixed TotalCharges column (text → float, 11 blank rows filled with 0)
- Dropped customerID (no predictive value)
- Encoded all categorical features (binary, 3-value, and one-hot)
- Scaled numerical features: tenure, MonthlyCharges, TotalCharges
- Trained Random Forest (100 trees, class_weight='balanced') on 80/20 stratified split

### Model performance (Random Forest)
| Metric | Score |
|--------|-------|
| Accuracy | 79.6% |
| F1 Score | 56.9% |
| ROC-AUC | 82.7% |

- Low F1 expected due to class imbalance (26.5% churn) — addressed with class_weight='balanced'
- Feature importances confirm EDA findings: tenure, MonthlyCharges and contract type are the strongest predictors

## Phase 3 — FastAPI Backend

Built a REST API with 4 endpoints:

| Endpoint | Description |
|---|---|
| `GET /` | Confirms API is running |
| `GET /health` | Health check for deployment |
| `POST /predict` | Takes customer data, returns churn prediction |
| `GET /features` | Returns top feature importance scores |

*In active development — updated weekly*

---

## Dataset
Telco Customer Churn available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)