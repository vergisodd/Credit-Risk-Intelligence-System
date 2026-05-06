# Credit Risk Intelligence System

End-to-end machine learning pipeline for predicting loan default risk using the Home Credit Default Risk dataset.

This project predicts whether a loan applicant is likely to experience payment difficulty and translates model outputs into practical business risk tiers for credit review.

---

## Project Objective

Lenders need to identify applicants who may be at higher risk of default while avoiding unnecessary rejection of reliable customers.

This project builds a credit risk workflow that:

- Cleans and prepares applicant data
- Engineers credit-risk features
- Trains baseline and advanced machine learning models
- Evaluates model performance using imbalance-aware metrics
- Compares Logistic Regression and XGBoost performance
- Converts predicted probabilities into business risk tiers
- Provides an interactive Streamlit app for applicant-level risk review and model comparison

---

## Business Problem

Credit risk models involve two major tradeoffs:

1. **False negatives**: risky applicants are approved and may later default.
2. **False positives**: reliable applicants are incorrectly flagged as risky, creating unnecessary manual review or lost lending opportunities.

Because the dataset is highly imbalanced, accuracy alone is misleading. This project focuses on ROC-AUC, recall, precision, F1-score, confusion matrix, and threshold analysis.

---

## Dataset

This project uses the Home Credit Default Risk dataset.

Current version uses:

```text
application_train.csv
```

Target variable:

```text
TARGET = 1 -> applicant had payment difficulty
TARGET = 0 -> applicant did not have payment difficulty
```

### Dataset Summary

| Item | Value |
|---|---:|
| Rows | 307,511 |
| Raw Columns | 122 |
| Final Model Features | 82 |
| Dropped Columns | 50 |
| Engineered Features | 11 |
| Default Rate | 8.07% |
| Non-Default Rate | 91.93% |

The raw dataset is not included in this repository because of file size and licensing considerations.

---

## Tools & Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Logistic Regression
- XGBoost
- Streamlit
- matplotlib
- seaborn
- joblib
- Git
- GitHub

---

## Project Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Preprocessing Pipeline
   ↓
Logistic Regression Baseline
   ↓
XGBoost Model
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Threshold Analysis
   ↓
Risk Tier Assignment
   ↓
Streamlit App
```

---

## Feature Engineering

The project adds domain-informed credit risk features:

| Feature | Meaning |
|---|---|
| `CREDIT_INCOME_RATIO` | Credit amount relative to applicant income |
| `ANNUITY_INCOME_RATIO` | Loan annuity burden relative to income |
| `GOODS_CREDIT_RATIO` | Goods price relative to credit amount |
| `CREDIT_TERM_RATIO` | Annuity relative to credit amount |
| `INCOME_PER_FAMILY_MEMBER` | Income adjusted by family size |
| `AGE_YEARS` | Applicant age |
| `EMPLOYMENT_YEARS` | Employment length |
| `EXT_SOURCE_MEAN` | Average external credit score |
| `EXT_SOURCE_MIN` | Minimum external credit score |
| `EXT_SOURCE_MAX` | Maximum external credit score |

---

## Models

This project currently compares two models:

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline linear classifier with class imbalance handling |
| XGBoost | Gradient boosting model for stronger predictive performance |

---

## Baseline Model Results

Baseline model used:

```text
LogisticRegression(class_weight="balanced")
```

Performance on the test set:

| Metric | Value |
|---|---:|
| Accuracy | 0.6900 |
| ROC-AUC | 0.7470 |
| Precision - Default Class | 0.1612 |
| Recall - Default Class | 0.6755 |
| F1 - Default Class | 0.2602 |

---

## XGBoost Model Results

XGBoost model used:

```text
XGBClassifier with scale_pos_weight for class imbalance
```

Performance on the test set:

| Metric | Value |
|---|---:|
| Accuracy | 0.7052 |
| ROC-AUC | 0.7613 |
| Precision - Default Class | 0.1689 |
| Recall - Default Class | 0.6763 |
| F1 - Default Class | 0.2703 |

---

## Model Comparison

| Metric | Logistic Regression | XGBoost | Better Model |
|---|---:|---:|---|
| Accuracy | 0.6900 | 0.7052 | XGBoost |
| ROC-AUC | 0.7470 | 0.7613 | XGBoost |
| Precision - Default Class | 0.1612 | 0.1689 | XGBoost |
| Recall - Default Class | 0.6755 | 0.6763 | XGBoost |
| F1 - Default Class | 0.2602 | 0.2703 | XGBoost |

XGBoost is the stronger model in this version. It improves ROC-AUC, accuracy, precision, recall, and F1-score compared with the Logistic Regression baseline.

The improvement is useful but modest. Precision for the default class is still low, so the model remains best suited for screening and manual review prioritization, not automatic loan rejection.

---

## Confusion Matrix Comparison

### Logistic Regression — Threshold 0.50

| Actual / Predicted | Predicted Non-Default | Predicted Default |
|---|---:|---:|
| Actual Non-Default | 39,080 | 17,458 |
| Actual Default | 1,611 | 3,354 |

### XGBoost — Threshold 0.50

| Actual / Predicted | Predicted Non-Default | Predicted Default |
|---|---:|---:|
| Actual Non-Default | 40,014 | 16,524 |
| Actual Default | 1,607 | 3,358 |

At the 0.50 threshold, XGBoost reduces false positives from 17,458 to 16,524 while keeping recall almost unchanged.

---

## Model Interpretation

The Logistic Regression baseline achieves a ROC-AUC of 0.7470, showing useful ranking ability.

The XGBoost model improves ROC-AUC to 0.7613, meaning it ranks applicants slightly better than the baseline model.

However, precision for the default class remains low. Many applicants flagged as risky are actually non-default applicants.

The models are better suited for:

- Risk screening
- Manual review prioritization
- Baseline benchmarking
- Credit risk analysis

They should not be used for automatic loan rejection.

---

## Threshold Analysis

Different thresholds create different business tradeoffs.

### Logistic Regression Threshold Summary

| Threshold | Recall | Precision | Business Meaning |
|---:|---:|---:|---|
| 0.20 | 0.9720 | 0.0908 | Very aggressive screening |
| 0.50 | 0.6755 | 0.1612 | Broad baseline screening |
| 0.70 | 0.3430 | 0.2554 | Stricter manual review queue |
| 0.80 | 0.1716 | 0.3403 | Conservative high-risk flag |

### XGBoost Threshold Summary

| Threshold | Recall | Precision | Business Meaning |
|---:|---:|---:|---|
| 0.20 | 0.9696 | 0.0931 | Very aggressive screening |
| 0.50 | 0.6763 | 0.1689 | Broad baseline screening |
| 0.70 | 0.3553 | 0.2709 | Stricter manual review queue |
| 0.80 | 0.1627 | 0.3732 | Conservative high-risk flag |

Lower thresholds catch more risky applicants but create more false positives. Higher thresholds reduce false positives but miss more defaults.

---

## Risk Tier Logic

| Risk Tier | Default Probability | Suggested Action |
|---|---:|---|
| Low Risk | `< 0.30` | Standard processing |
| Medium Risk | `0.30 - 0.59` | Additional review if loan amount is high |
| High Risk | `>= 0.60` | Manual risk review recommended |

---

## Streamlit App

The project includes an interactive Streamlit app that allows users to:

- View project summary
- Select applicant IDs
- Generate default-risk predictions
- View risk tiers
- Review actual historical outcomes
- Compare Logistic Regression and XGBoost performance
- Inspect model performance
- Explore threshold analysis

Run locally:

```bash
streamlit run app/streamlit_app.py
```

---

## Repository Structure

```text
Credit-Risk-Intelligence-System/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── models/
│   └── .gitkeep
├── notebooks/
│   └── 01_eda_baseline.ipynb
├── reports/
│   ├── baseline_model_metrics.json
│   ├── business_recommendations.md
│   ├── logistic_regression_evaluation.json
│   ├── model_card.md
│   ├── model_comparison.md
│   ├── sample_predictions.csv
│   ├── threshold_analysis.csv
│   ├── xgboost_model_metrics.json
│   └── xgboost_threshold_analysis.csv
├── src/
│   ├── data_cleaning.py
│   ├── evaluate_model.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── train_model.py
│   └── train_xgboost.py
├── visuals/
│   ├── confusion_matrix_threshold_0_50.png
│   ├── confusion_matrix_xgboost_threshold_0_50.png
│   ├── roc_curve_logistic_regression.png
│   └── roc_curve_xgboost.png
├── .gitignore
├── README.md
└── requirements.txt
```

---

## How to Run This Project

### 1. Clone the repository

```bash
git clone https://github.com/vergisodd/Credit-Risk-Intelligence-System.git
cd Credit-Risk-Intelligence-System
```

### 2. Create a virtual environment

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add dataset

Download the Home Credit Default Risk dataset and place:

```text
application_train.csv
```

inside:

```text
data/raw/
```

### 5. Train the Logistic Regression baseline

```bash
python src/train_model.py
```

### 6. Train the XGBoost model

```bash
python src/train_xgboost.py
```

### 7. Evaluate the Logistic Regression model

```bash
python src/evaluate_model.py
```

### 8. Generate sample predictions

```bash
python src/predict.py
```

### 9. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## Key Project Files

| File | Purpose |
|---|---|
| `src/data_cleaning.py` | Loads data, handles missing columns, prepares features and target |
| `src/feature_engineering.py` | Adds domain-specific credit risk features |
| `src/train_model.py` | Trains the Logistic Regression baseline model |
| `src/train_xgboost.py` | Trains the XGBoost credit risk model |
| `src/evaluate_model.py` | Evaluates Logistic Regression and saves threshold analysis |
| `src/predict.py` | Generates applicant-level risk predictions |
| `app/streamlit_app.py` | Interactive dashboard for risk review and model comparison |
| `reports/model_card.md` | Technical model documentation |
| `reports/model_comparison.md` | Compares Logistic Regression and XGBoost performance |
| `reports/business_recommendations.md` | Business interpretation and recommendations |

---

## Current Limitations

This is a strong baseline project, but it is not a production lending system.

Main limitations:

- Uses only `application_train.csv`
- Does not yet use bureau, previous application, installment, or credit card history tables
- Logistic Regression and XGBoost still have low precision for the default class
- XGBoost has been added, but Random Forest has not been added yet
- No SHAP explainability yet
- No full fairness or bias analysis yet
- No deployed public app yet
- No model monitoring or drift simulation yet

---

## Next Improvements

Planned next steps:

- Add Random Forest as a third comparison model
- Add SHAP explainability
- Tune thresholds using business cost assumptions
- Add feature importance analysis
- Integrate additional Home Credit relational tables
- Deploy Streamlit app publicly
- Add model monitoring and drift simulation
- Add screenshots of the Streamlit app to the README

---

## Business Takeaway

The models are not strong enough for automatic lending decisions, but they are useful as credit risk screening tools.

The strongest current use case is helping credit teams prioritize manual review by ranking applicants based on predicted payment difficulty risk.

XGBoost is currently the best-performing model in the project, but its improvement over Logistic Regression is modest. The project should be treated as a realistic credit risk analysis system, not a production-grade automated lending engine.
