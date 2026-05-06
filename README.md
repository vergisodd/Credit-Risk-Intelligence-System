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
