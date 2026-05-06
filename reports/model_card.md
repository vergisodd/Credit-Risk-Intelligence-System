# Model Card — Logistic Regression Baseline

## Project

Credit Risk Intelligence System

## Model Name

Logistic Regression Baseline

## Model Type

Binary classification model

## Objective

The model predicts whether a loan applicant is likely to experience payment difficulty.

Target variable:

- `TARGET = 1`: applicant had payment difficulty
- `TARGET = 0`: applicant did not have payment difficulty

## Dataset

This model uses the Home Credit Default Risk training dataset, specifically:

- `application_train.csv`

The raw dataset contains:

- 307,511 rows
- 122 columns

After cleaning and feature selection:

- 82 final features were used
- 50 high-missing or non-predictive columns were dropped
- 11 engineered domain features were added

## Feature Engineering

The following domain-informed features were created:

- `CREDIT_INCOME_RATIO`
- `ANNUITY_INCOME_RATIO`
- `GOODS_CREDIT_RATIO`
- `CREDIT_TERM_RATIO`
- `INCOME_PER_FAMILY_MEMBER`
- `AGE_YEARS`
- `DAYS_EMPLOYED_CLEAN`
- `EMPLOYMENT_YEARS`
- `EXT_SOURCE_MEAN`
- `EXT_SOURCE_MIN`
- `EXT_SOURCE_MAX`

These features help capture applicant affordability, repayment burden, age, employment history, and external credit score patterns.

## Data Preprocessing

Numeric features:

- Missing values imputed using median
- Features standardized using `StandardScaler`

Categorical features:

- Missing values imputed using most frequent value
- Encoded using one-hot encoding

Columns with more than 40% missing values were removed for the baseline model.

## Train/Test Split

The dataset was split using:

- 80% training data
- 20% testing data
- Stratified split to preserve target imbalance
- Random state: 42

## Class Imbalance

The target variable is highly imbalanced:

- Non-default/payment difficulty class: 91.93%
- Default/payment difficulty class: 8.07%

Because of this imbalance, accuracy is not the main evaluation metric.

## Evaluation Metrics

Baseline model performance on the test set:

| Metric | Value |
|---|---:|
| Accuracy | 0.6900 |
| ROC-AUC | 0.7470 |
| Precision — Default Class | 0.1612 |
| Recall — Default Class | 0.6755 |
| F1 — Default Class | 0.2602 |

## Confusion Matrix

At classification threshold 0.50:

|  | Predicted Non-Default | Predicted Default |
|---|---:|---:|
| Actual Non-Default | 39,080 | 17,458 |
| Actual Default | 1,611 | 3,354 |

## Interpretation

The Logistic Regression baseline achieved a ROC-AUC score of 0.7470, which shows that the model has useful ranking ability.

However, the model has low precision for the default class. This means that many applicants flagged as risky are actually non-default applicants.

The model has stronger recall than precision, meaning it is better at catching risky applicants than making highly accurate high-risk predictions.

## Threshold Analysis Summary

Different classification thresholds create different business tradeoffs:

| Threshold | Business Meaning |
|---|---|
| 0.20 | Very aggressive risk screening; catches most defaults but creates many false positives |
| 0.50 | Balanced baseline threshold |
| 0.70 | Stricter manual review threshold |
| 0.80 | Very conservative high-risk flag; fewer false positives but misses many defaults |

## Recommended Use

This baseline model should not be used for automatic loan rejection.

A better use case would be:

- Risk screening
- Manual review prioritization
- Early warning analysis
- Model benchmarking before training stronger models

## Limitations

This is a baseline model and has several limitations:

- Low precision for the default class
- High false positive count
- Uses only `application_train.csv`
- Does not yet use bureau, previous application, installment, or credit card history tables
- Does not include advanced hyperparameter tuning
- Does not include full fairness or bias testing

## Next Steps

Recommended improvements:

1. Train Random Forest and XGBoost models
2. Tune classification thresholds based on business cost
3. Add SHAP explainability
4. Include additional Home Credit relational tables
5. Create a Streamlit app for business users
6. Add model monitoring and drift analysis