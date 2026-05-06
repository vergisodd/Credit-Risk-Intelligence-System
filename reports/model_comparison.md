# Model Comparison Report

## Project

Credit Risk Intelligence System

## Objective

This report compares the baseline Logistic Regression model against an XGBoost model for loan default risk prediction.

The goal is to evaluate whether a more advanced tree-based model improves predictive performance on the Home Credit Default Risk dataset.

---

## Models Compared

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier with `class_weight="balanced"` |
| XGBoost | Gradient boosting classifier with class imbalance weighting using `scale_pos_weight` |

---

## Test Set Performance

| Metric | Logistic Regression | XGBoost | Better Model |
|---|---:|---:|---|
| Accuracy | 0.6900 | 0.7052 | XGBoost |
| ROC-AUC | 0.7470 | 0.7613 | XGBoost |
| Precision — Default Class | 0.1612 | 0.1689 | XGBoost |
| Recall — Default Class | 0.6755 | 0.6763 | XGBoost |
| F1 — Default Class | 0.2602 | 0.2703 | XGBoost |

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

---

## Key Findings

### 1. XGBoost improves ROC-AUC

XGBoost achieved a ROC-AUC of 0.7613 compared to 0.7470 for Logistic Regression.

This means XGBoost has better ranking ability and is better at separating higher-risk applicants from lower-risk applicants.

### 2. XGBoost reduces false positives

At the 0.50 threshold:

- Logistic Regression false positives: 17,458
- XGBoost false positives: 16,524

XGBoost reduced false positives by 934 applicants.

This matters because false positives can create unnecessary manual reviews and may incorrectly flag reliable applicants as risky.

### 3. Recall is almost unchanged

Default-class recall:

- Logistic Regression: 0.6755
- XGBoost: 0.6763

XGBoost catches roughly the same share of risky applicants as Logistic Regression.

### 4. Precision is still low

XGBoost precision for the default class is 0.1689.

This is better than Logistic Regression, but still low. Many applicants flagged as risky are still non-default applicants.

This means the model should not be used for automatic loan rejection.

---

## Business Interpretation

XGBoost is the stronger model for this version of the project.

It improves ROC-AUC, accuracy, precision, recall, F1-score, and reduces false positives.

However, the improvement is modest. The model is useful for risk screening and manual review prioritization, not automatic lending decisions.

---

## Recommended Model

The recommended model for the current version is:

```text
XGBoost Credit Risk Model