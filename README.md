# Credit Risk Intelligence System

An end-to-end machine learning project for predicting loan default risk using applicant financial and demographic data.

## Project Objective

The goal of this project is to predict whether a loan applicant is likely to experience payment difficulty.

This project goes beyond building a model. It focuses on turning model predictions into practical business risk tiers that could support credit review, loan approval decisions, and manual risk investigation.

## Business Problem

Lenders need to identify applicants who may be at higher risk of default while avoiding unnecessary rejection of reliable customers.

A poor credit risk model can create two major problems:

- Approving applicants who are likely to default
- Rejecting applicants who would have repaid successfully

This project uses machine learning to support better risk-based decision-making.

## Dataset

This project uses the Home Credit Default Risk dataset.

The first version of the project focuses only on:

```text
application_train.csv

Target variable:

TARGET = 1 → client had payment difficulty
TARGET = 0 → client did not have payment difficulty

The raw dataset is not included in this repository because of file size and licensing considerations.

Tools & Technologies
Python
pandas
NumPy
scikit-learn
XGBoost
SHAP
Streamlit
matplotlib
seaborn
Git & GitHub
Project Workflow
Data loading
Exploratory data analysis
Data cleaning
Feature engineering
Baseline model training
Advanced model training
Model evaluation
Model explainability
Business risk tier creation
Streamlit app deployment
Models

The project will compare multiple models:

Logistic Regression
Random Forest
XGBoost
Evaluation Metrics

Because credit risk datasets are usually imbalanced, accuracy alone is not enough.

The project will evaluate models using:

ROC-AUC
Precision
Recall
F1-score
Confusion matrix
Business Output

The final system will classify applicants into risk tiers:

Low Risk
Medium Risk
High Risk

The goal is not only to predict risk, but also to explain the main factors behind each prediction.

Repository Structure
Credit-Risk-Intelligence-System/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── 01_eda_baseline.ipynb
├── reports/
│   ├── business_recommendations.md
│   └── model_card.md
├── src/
│   ├── data_cleaning.py
│   ├── evaluate_model.py
│   ├── feature_engineering.py
│   ├── predict.py
│   └── train_model.py
├── visuals/
├── .gitignore
├── README.md
└── requirements.txt