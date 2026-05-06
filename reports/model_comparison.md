# Model Comparison Report

## Summary

The strongest holdout model is **LightGBM** with ROC-AUC 0.7715 and Average Precision 0.2608.

| Model | AUC-ROC | Average Precision | Tuned CV AUC | F1-Default | Precision-Default | Recall-Default | Optimal Threshold | FP at Optimal | FN at Optimal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7507 | 0.2333 |  | 0.2623 | 0.1626 | 0.6777 | 0.6500 | 7384 | 2782 |
| XGBoost | 0.7680 | 0.2575 | 0.7618 | 0.2877 | 0.1855 | 0.6401 | 0.6800 | 4897 | 3086 |
| LightGBM | 0.7715 | 0.2608 | 0.7655 | 0.2795 | 0.1754 | 0.6878 | 0.6600 | 6498 | 2762 |

## Calibration Interpretation

The calibration plot compares predicted default probabilities with observed default rates across probability bins. For risk-tiering, calibration matters because a score near 0.59 should behave like a materially higher-risk applicant group than a score near 0.30. If the curve sits far from the diagonal, predicted probabilities are still useful for ranking, but the exact percentages should be treated as review scores rather than literal default-rate estimates.
