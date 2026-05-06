# Model Card

## Model Version

LightGBM+Bureau (Optuna-tuned) is the strongest portfolio model. LightGBM (Optuna-tuned), XGBoost, and Logistic Regression are retained as benchmark models for comparison and governance review.

## Intended Use

Manual review queue prioritization. A human analyst must make all final credit decisions. Not suitable for automated decisioning.

## Training Data

The system uses `application_train.csv` from the Home Credit Default Risk dataset, with optional bureau credit-history aggregations from `bureau.csv`. Models are trained and evaluated using a stratified 80/20 train/test split, with 5-fold cross-validation for tuned model reporting.

## Performance

| Model | Test AUC | Average Precision | Tuned CV AUC (5-fold) | Tuned CV AUC Std |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7507 | 0.2333 | N/A | N/A |
| XGBoost | 0.7680 | 0.2575 | 0.7618 | 0.0008 |
| LightGBM | 0.7715 | 0.2608 | 0.7655 | 0.0011 |
| LightGBM+Bureau | 0.7746 | 0.2681 | 0.7702 | 0.0010 |

## Primary Model Metrics

- Model version: LightGBM (Optuna-tuned), XGBoost, Logistic Regression (baseline)
- Tuned CV AUC (5-fold): 0.7655
- Tuned CV AUC std: 0.0011
- Test AUC: 0.7715
- Average Precision: 0.2608
- No temporal validation: performance on future applicant cohorts is unknown.

## Fairness

Equalized Odds gap by gender = 0.2371 at lender-cost-optimal threshold. AUC degradation from removing `CODE_GENDER` = 0.0017. See `reports/fairness_report.md`.

## Regulatory Note

This system is NOT intended for use as an automated credit decision engine. Any deployment in a regulated lending context requires compliance review under ECOA (US), GDPR Article 22 (EU), and the EU AI Act high-risk classification for credit scoring. The fairness analysis in this repository is diagnostic only.

## Known Limitations

This project is a portfolio demonstration, not a production lending system.

**Data scope:** Only `application_train.csv` is used as the primary feature source. Bureau credit history is integrated via `src/feature_engineering_bureau.py` (13 aggregated features). The remaining four relational tables — `previous_application`, `installments_payments`, `POS_CASH_balance`, and `credit_card_balance` — are not yet integrated. Full integration is the primary path to closing the remaining gap against leaderboard AUC benchmarks.

**Modelling:** Precision for the default class remains low (approximately 0.17-0.18 at threshold 0.50), which is expected given an 8% base rate. At operating thresholds aligned to business cost scenarios (threshold 0.20-0.35), recall is high but precision remains low. This reflects the fundamental difficulty of the problem rather than a modelling failure.

**Explainability:** SHAP values describe model behavior, not causal credit risk factors. Individual applicant explanations show which features contributed most to a specific prediction — they do not constitute a regulatory explanation of an adverse action.

**Fairness:** The model exhibits a 0.24 Equalized Odds gap between male and female applicants at the lender-cost-optimal threshold. Removing `CODE_GENDER` reduces AUC by 0.0017. This tradeoff requires governance review before any regulated deployment. The fairness analysis in this project is diagnostic, not a mitigation.

**Deployment:** This system is not suitable for automated credit decisioning. It is designed for manual review queue prioritization only, with a human analyst making the final credit determination.
