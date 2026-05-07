# Model Comparison Report

## Summary

The champion model is **LightGBM+Bureau** (`application+bureau` feature set). The strongest holdout model in the comparison is **LightGBM+Bureau** with ROC-AUC 0.7746 and Average Precision 0.2681.

| Model | AUC-ROC | Average Precision | Tuned CV AUC | F1-Default | Precision-Default | Recall-Default | F1-Optimal Threshold | F1 at F1-Optimal | Cost-Min Threshold | Min Relative Cost | Precision-Selected | Recall-Selected | Review Volume-Selected | Missed Defaults-Selected | Review Volume-Cost-Min | Missed Defaults-Cost-Min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7507 | 0.2333 |  | 0.2623 | 0.1626 | 0.6777 | 0.6500 | 0.3004 | 0.5400 | 33288.0000 | 0.2282 | 0.4397 | 9567 | 2782 | 17364 | 1899 |
| XGBoost | 0.7680 | 0.2575 | 0.7618 | 0.2877 | 0.1855 | 0.6401 | 0.6800 | 0.3201 | 0.4900 | 31748.0000 | 0.2773 | 0.3784 | 6776 | 3086 | 17848 | 1715 |
| LightGBM | 0.7715 | 0.2608 | 0.7655 | 0.2795 | 0.1754 | 0.6878 | 0.6600 | 0.3224 | 0.5000 | 31557.0000 | 0.2532 | 0.4437 | 8701 | 2762 | 19472 | 1550 |
| LightGBM+Bureau | 0.7746 | 0.2681 | 0.7702 | 0.2826 | 0.1783 | 0.6810 | 0.6600 | 0.3250 | 0.5300 | 31225.0000 | 0.2549 | 0.4483 | 8732 | 2739 | 16709 | 1771 |

LightGBM+Bureau achieved +0.0032 AUC improvement from bureau feature integration


## Threshold Definitions

`default_threshold` is the conventional 0.50 classifier cutoff.

`cost_minimizing_threshold` minimizes a stated false-negative/false-positive cost scenario. In the bundled lender scenario, false negatives are weighted 10x false positives.

`f1_optimal_threshold` maximizes default-class F1. This is the configured operating threshold for the portfolio review queue because it balances precision and recall for manual review prioritization.

`risk_tiers` are score bands used for analyst triage and are not the same thing as a binary classifier threshold.

## Calibration Interpretation

The calibration plot compares predicted default probabilities with observed default rates across probability bins. For risk-tiering, calibration matters because a score near 0.59 should behave like a materially higher-risk applicant group than a score near 0.30. If the curve sits far from the diagonal, predicted probabilities are still useful for ranking, but the exact percentages should be treated as review scores rather than literal default-rate estimates.
