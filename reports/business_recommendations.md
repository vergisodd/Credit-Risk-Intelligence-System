# Business Recommendations — Credit Risk Intelligence System

## Executive Summary

The baseline credit risk model shows useful predictive power for identifying applicants who may experience payment difficulty.

The model achieved a ROC-AUC score of 0.7470, meaning it can rank higher-risk applicants better than random guessing.

However, the model should not be used for automatic loan rejection because precision for the default class is low. The better business use case is risk screening and manual review prioritization.

## Key Findings

### 1. The dataset is highly imbalanced

Only 8.07% of applicants in the dataset experienced payment difficulty.

This means a model could appear accurate by predicting most applicants as non-default. For this reason, accuracy alone is not a reliable success metric.

More useful metrics include:

- ROC-AUC
- Recall
- Precision
- F1-score
- Confusion matrix

### 2. The baseline model catches many risky applicants

At a 0.50 classification threshold, the model identified:

- 3,354 true default/payment difficulty cases
- 1,611 missed default/payment difficulty cases

This gives the model a default-class recall of 67.55%.

In business terms, the model catches about two-thirds of risky applicants in the test set.

### 3. The model creates many false positives

At the same 0.50 threshold, the model incorrectly flagged:

- 17,458 non-default applicants as risky

This is the biggest weakness of the baseline model.

A high false-positive count could create unnecessary manual reviews, customer friction, and lost lending opportunities.

### 4. Threshold selection matters

The model’s predicted probabilities can be converted into business risk tiers instead of relying only on a single default threshold.

Suggested risk tiers:

| Risk Tier | Probability Range | Suggested Action |
|---|---:|---|
| Low Risk | Less than 0.30 | Standard processing |
| Medium Risk | 0.30 to 0.59 | Additional review if loan amount is high |
| High Risk | 0.60 and above | Manual risk review |

This makes the model more practical for business decision-making.

## Business Recommendation

The model should be used as a decision-support tool, not as an automatic approval or rejection system.

Recommended workflow:

1. Low-risk applicants continue through standard review.
2. Medium-risk applicants receive additional checks if the credit amount is large.
3. High-risk applicants are routed to manual review.
4. Final decisions remain with a human credit analyst.

## Why This Matters

The model helps lenders prioritize review effort.

Instead of manually reviewing every applicant equally, the lender can focus attention on applicants with higher predicted risk.

This can improve operational efficiency while reducing the chance of missing risky applications.

## Current Model Limitations

The current Logistic Regression baseline has useful signal, but it is not final.

Main limitations:

- Low precision for default predictions
- High false-positive count
- Only one dataset table used
- No hyperparameter tuning yet
- No advanced model comparison yet
- No SHAP explainability yet
- No fairness testing yet

## Next Technical Steps

The next version should improve the model by adding:

1. Random Forest model
2. XGBoost model
3. SHAP explainability
4. Cost-based threshold selection
5. More Home Credit relational tables
6. Streamlit dashboard for interactive review

## Final Business Positioning

This project demonstrates how machine learning can support credit risk analysis by combining:

- Predictive modelling
- Risk tiering
- Model evaluation
- Business interpretation

The strongest current use case is not automatic decision-making. The strongest use case is helping credit teams prioritize manual review and identify applicants who deserve closer investigation.