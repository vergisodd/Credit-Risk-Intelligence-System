# Credit Risk Intelligence System

End-to-end machine learning pipeline for predicting loan default risk using the Home Credit Default Risk dataset.

This project predicts whether a loan applicant is likely to experience payment difficulty and translates model outputs into practical business risk tiers for credit review.

---

## Project Objective

Lenders need to identify applicants who may be at higher risk of default while avoiding unnecessary rejection of reliable customers.

This project builds a credit risk prediction workflow that:

- Cleans and prepares applicant data
- Engineers credit-risk features
- Trains a baseline machine learning model
- Evaluates model performance using imbalance-aware metrics
- Converts predicted probabilities into business risk tiers
- Provides an interactive Streamlit app for applicant-level risk review

---

## Business Problem

Credit risk models create two major business tradeoffs:

1. **False negatives**  
   Risky applicants are approved and may later default.

2. **False positives**  
   Reliable applicants are incorrectly flagged as risky, creating unnecessary manual review or lost lending opportunities.

Because the dataset is highly imbalanced, accuracy alone is misleading. The project focuses on ROC-AUC, recall, precision, F1-score, confusion matrix, and threshold analysis.

---

## Dataset

This project uses the Home Credit Default Risk dataset.

Current version uses:

```text
application_train.csv

Target variable:

TARGET = 1 → applicant had payment difficulty
TARGET = 0 → applicant did not have payment difficulty

Dataset summary:

Item	Value
Rows	307,511
Raw Columns	122
Final Model Features	82
Dropped Columns	50
Engineered Features	11
Default Rate	8.07%
Non-Default Rate	91.93%

The raw dataset is not included in this repository because of file size and licensing considerations.

Tools & Technologies
Python
pandas
NumPy
scikit-learn
Logistic Regression
Streamlit
matplotlib
seaborn
joblib
Git & GitHub
Project Workflow
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
Model Evaluation
   ↓
Threshold Analysis
   ↓
Risk Tier Assignment
   ↓
Streamlit App
Feature Engineering

The project adds domain-informed credit risk features:

Feature	Meaning
CREDIT_INCOME_RATIO	Credit amount relative to applicant income
ANNUITY_INCOME_RATIO	Loan annuity burden relative to income
GOODS_CREDIT_RATIO	Goods price relative to credit amount
CREDIT_TERM_RATIO	Annuity relative to credit amount
INCOME_PER_FAMILY_MEMBER	Income adjusted by family size
AGE_YEARS	Applicant age
EMPLOYMENT_YEARS	Employment length
EXT_SOURCE_MEAN	Average external credit score
EXT_SOURCE_MIN	Minimum external credit score
EXT_SOURCE_MAX	Maximum external credit score
Baseline Model Results

Model used:

Logistic Regression with class_weight="balanced"

Performance on the test set:

Metric	Value
Accuracy	0.6900
ROC-AUC	0.7470
Precision — Default Class	0.1612
Recall — Default Class	0.6755
F1 — Default Class	0.2602
Confusion Matrix

At threshold 0.50:

	Predicted Non-Default	Predicted Default
Actual Non-Default	39,080	17,458
Actual Default	1,611	3,354
Model Interpretation

The baseline model achieves a ROC-AUC of 0.7470, which shows useful ranking ability.

However, precision for the default class is low. This means many applicants flagged as risky are actually non-default applicants.

The model is better suited for:

Risk screening
Manual review prioritization
Baseline benchmarking
Credit risk analysis

It should not be used for automatic loan rejection.

Threshold Analysis

Different thresholds create different business tradeoffs.

Threshold	Recall	Precision	Business Meaning
0.20	0.9720	0.0908	Very aggressive screening
0.50	0.6755	0.1612	Broad baseline screening
0.70	0.3430	0.2554	Stricter manual review queue
0.80	0.1716	0.3403	Conservative high-risk flag

Lower thresholds catch more risky applicants but create more false positives. Higher thresholds reduce false positives but miss more defaults.

Risk Tier Logic
Risk Tier	Default Probability	Suggested Action
Low Risk	< 0.30	Standard processing
Medium Risk	0.30 – 0.59	Additional review if loan amount is high
High Risk	>= 0.60	Manual risk review recommended
Streamlit App

The project includes an interactive Streamlit app that allows users to:

View project summary
Select applicant IDs
Generate default-risk predictions
View risk tiers
Review actual historical outcomes
Inspect model performance
Explore threshold analysis

Run locally:

streamlit run app/streamlit_app.py
Repository Structure
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
│   ├── sample_predictions.csv
│   └── threshold_analysis.csv
├── src/
│   ├── data_cleaning.py
│   ├── evaluate_model.py
│   ├── feature_engineering.py
│   ├── predict.py
│   └── train_model.py
├── visuals/
│   ├── confusion_matrix_threshold_0_50.png
│   └── roc_curve_logistic_regression.png
├── .gitignore
├── README.md
└── requirements.txt
