# Model Card

## Champion Model

**LightGBM+Bureau** is the configured champion model. The artifact registry is defined in `config.yaml` and summarized in `reports/model_manifest.json`.

Benchmark models remain in the repository for comparison: Logistic Regression, XGBoost, and application-only LightGBM.

## Intended Use

Manual credit-risk review queue prioritization. A human analyst must make all final credit decisions. The model must not be used for automatic approval or rejection.

## Training Data

The champion uses the Home Credit `application_train.csv` file plus 13 applicant-level bureau aggregations from `bureau.csv`. Training uses a stratified 80/20 train/test split with 5-fold cross-validation for tuned model reporting.

Required local files:

| File | Purpose |
|---|---|
| `data/raw/application_train.csv` | Primary applicant features and target |
| `data/raw/bureau.csv` | Required for the champion `application+bureau` feature set |

## Performance

| Model | Test ROC-AUC | Average Precision | Tuned CV AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.7507 | 0.2333 | N/A |
| XGBoost | 0.7680 | 0.2575 | 0.7618 |
| LightGBM | 0.7715 | 0.2608 | 0.7655 |
| **LightGBM+Bureau** | **0.7746** | **0.2681** | **0.7702** |

The target base rate is approximately 8.07%, so Average Precision should be interpreted relative to a difficult imbalanced-class baseline rather than as a standalone percentage.

## Threshold Policy

Threshold names are explicit:

| Name | Value | Meaning |
|---|---:|---|
| Default threshold | 0.50 | Conventional classifier cutoff |
| F1-optimal threshold | 0.66 | Configured operating point for review queue prioritization |
| Lender cost-minimizing threshold | 0.53 | Cost-policy result under FN cost 10x FP cost |
| Risk tiers | 0.30 / 0.59 | Low, medium, and high analyst queue bands |

Fairness reports must use and label the selected threshold explicitly. The F1-optimal threshold is not called lender-cost-optimal.

## Fairness

The fairness workflow computes subgroup ROC-AUC, Average Precision, false positive rate, false negative rate, predicted default rate, and actual default rate for `CODE_GENDER`, `NAME_EDUCATION_TYPE`, and `ORGANIZATION_TYPE`.

Sensitive variables are retained in this portfolio experiment for transparency. Regulated use would require legal, compliance, and model-risk review. Removing protected attributes does not remove proxy bias, and the fairness metrics here are diagnostics rather than mitigation.

## Explainability

Global and applicant-level SHAP are generated for the champion model. One-hot encoded variables are aggregated back to source feature names where practical. Applicant reason codes show positive contributors increasing risk, negative contributors decreasing risk, and raw applicant values where available.

SHAP describes model behavior, not causality, and is not a legally sufficient adverse-action explanation.

## Known Limitations

This is a portfolio demonstration, not a production lending system. It does not include temporal validation, drift monitoring, legal policy rules, adverse-action workflows, or all Home Credit relational tables. Scores should be treated as review-prioritization signals, not calibrated probabilities or automated credit decisions.
