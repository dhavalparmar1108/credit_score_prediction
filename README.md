# Credit Score Movement Prediction

This project predicts customer credit score movements — whether it is likely to **Increase**, **Decrease**, or remain **Stable** — using machine learning models trained on synthetic financial and behavioral data.

---

## Problem Statement

Financial institutions need early indicators of creditworthiness changes to reduce risk and improve portfolio quality. Traditional credit scoring doesn't offer dynamic predictions. This project aims to predict **credit score trends**, not static scores.

---

## Dataset Overview

The dataset contains features such as:

- Demographics: `age`, `gender`, `location`
- Financials: `monthly_income`, `monthly_emi_outflow`, `emi_to_income_ratio`, `current_outstanding`, `total_credit_limit`
- Behavioral: `dpd_last_3_months`, `months_since_last_default`, `num_hard_inquiries_last_6`, `credit_utilization_ratio`, `repayment_history_score`
- Usage: `recent_credit_card_usage`, `recent_loan_disbursed_amount`
- **Target Variable**: `target_credit_score_movement` (`Increase`, `Decrease`, `Stable`)

> The target was generated using domain-inspired, **probabilistic rules** to avoid information leakage and hard overfitting by models.

---

## Project Flow

1. **Dataset Creation**:  
   Open `create_dataset.ipynb` to generate a realistic, rule-based synthetic dataset. This notebook exports a `.csv` file.

2. **Model Training**:  
   Load the exported dataset in `model.ipynb`, perform preprocessing, apply sampling, and train various models.

---

## Model Building

### Preprocessing
- Null imputation
- Outlier detection & treatment
- Feature engineering:
  - `emi_to_income_ratio`
  - `never_defaulted` (from `months_since_last_default`)
- Dropped directly correlated features to avoid leakage
- Applied **SMOTE** and **ADASYN** for balancing

### Models Tried
- Random Forest Classifier *(Best performing)*
- Logistic Regression *(Baseline, underfitted)*
- XGBoost Classifier
- VotingClassifier (Ensemble: XGB + RF + LogisticRegression)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Stratified Train-Test Splitting for stable representation

---

## Handling Class Imbalance

- Oversampling using:
  - `SMOTE`
  - `ADASYN`
  - `KMeansSMOTE`
- Class weights passed into models
- Evaluation done on original distribution to simulate real-world skew

---

## Final Model Results (Random Forest Example)

- Accuracy: **0.65**
- Best Cross-Validation Score: **0.7898**
- F1 Score (Weighted): **~0.61**

> Class `Decrease` still underperforms due to inherent rule imbalance and complex overlap in features.

---

## Business Takeaways

- **High EMI-to-Income Ratio & Recent DPDs** → Likely score *Decrease* → High-risk segment
- **Low Credit Utilization + Good Repayment** → Score *Increase* → Ideal for cross-sell opportunities
- **Hard Inquiries + Overleveraged Usage** → Early warning signs → Trigger preemptive retention actions
- Credit trend predictions allow **proactive portfolio interventions** rather than reactive decisions

---

## Tech Stack

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy`, `scikit-learn`, `xgboost`
  - `imbalanced-learn`: SMOTE, ADASYN, KMeansSMOTE
  - `matplotlib`, `seaborn` for visualization
- **Environment**: Jupyter Notebook

---

## File Structure

```bash
.
├── create_dataset.ipynb        # Synthetic dataset generator
├── model.ipynb                 # Model training, evaluation, and insights
├── credit_score_movement.csv   # Generated dataset
├── README.md                   # Project documentation
