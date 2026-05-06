# Model Card

## Model Family

LightGBM is the primary model. XGBoost is the secondary gradient-boosted benchmark, and Logistic Regression is the baseline classifier.

## Training Data

The system uses `application_train.csv` from the Home Credit Default Risk dataset. Models are trained with a stratified 80/20 train/test split and 5-fold cross-validation where configured.

## Performance

| Model | Holdout AUC | Average Precision | CV AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.7507 | 0.2333 | N/A |
| XGBoost | 0.7680 | 0.2575 | 0.7618 |
| LightGBM | 0.7715 | 0.2608 | 0.7491 |

## Fairness Evaluation

Fairness results are documented in [fairness_report.md](fairness_report.md). The M/F Equalized Odds gap is 0.2371 at the F1-optimal LightGBM threshold of 0.66.

## Regulatory Note

This system is NOT intended for use as an automated credit decision engine. Any deployment in a regulated lending context requires compliance review under ECOA (US), GDPR Article 22 (EU), and equivalent national frameworks.

## Known Limitations

- Low precision for the default class
- High false-positive volume at recall-oriented thresholds
- Uses only `application_train.csv`
- Does not use bureau, previous application, installment, or credit card history tables
- No temporal validation - performance on future applicant cohorts is unknown
- SHAP values describe model contribution, not causal credit risk factors
- Sensitive attributes require governance review before deployment

## Recommended Use

Manual review prioritization only. The model can help rank applicants and support analyst investigation, but it should not approve, deny, or price credit automatically.
