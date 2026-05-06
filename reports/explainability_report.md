# Explainability Report

## Global SHAP Findings

Global SHAP analysis was computed on 1,000 stratified holdout applicants using the trained LightGBM pipeline. The top global features are:

| Rank | Feature | Direction | Mean Abs SHAP | Interpretation |
|---:|---|---|---:|---|
| 1 | `EXT_SOURCE_MEAN` | decreases risk on average | 0.380033 | Lower external-source scores generally indicate a weaker external credit-risk signal. |
| 2 | `CREDIT_TERM_RATIO` | increases risk on average | 0.185655 | A higher annuity-to-credit relationship can indicate a shorter or heavier repayment structure. |
| 3 | `CODE_GENDER` | increases risk on average | 0.134418 | Gender is a sensitive demographic attribute and requires governance review before any deployment. |
| 4 | `EXT_SOURCE_3` | decreases risk on average | 0.132076 | This external score often captures repayment-risk signal not visible in simple affordability ratios. |
| 5 | `GOODS_CREDIT_RATIO` | decreases risk on average | 0.116276 | This compares the financed goods price with the total requested credit. |
| 6 | `NAME_EDUCATION_TYPE` | increases risk on average | 0.108848 | Education category may proxy socioeconomic patterns and should be reviewed for fairness impact. |
| 7 | `EXT_SOURCE_MAX` | increases risk on average | 0.092381 | The strongest external-source score can offset some weaker application attributes. |
| 8 | `FLAG_OWN_CAR` | decreases risk on average | 0.090737 | This feature contributes predictive signal in the model but should not be interpreted as causal evidence. |
| 9 | `AMT_ANNUITY` | increases risk on average | 0.090394 | Higher required annuity payments can increase repayment burden. |
| 10 | `AMT_GOODS_PRICE` | increases risk on average | 0.085415 | The goods price helps contextualize the requested credit amount. |

## Fairness Note

`CODE_GENDER` and `NAME_EDUCATION_TYPE` appear in the feature set and may contribute to model predictions. These variables require fairness review because they can directly encode or proxy protected and socioeconomic characteristics.

## Per-Applicant Explanation Methodology

Applicant-level explanations transform the applicant row through the fitted preprocessing pipeline, calculate LightGBM SHAP values on the transformed feature vector, and render a waterfall plot showing the strongest positive and negative contributors to that specific prediction.

## Limitations

Global feature importance reflects aggregate model behavior across the holdout population. SHAP values for individual applicants indicate feature contribution to that specific prediction, not causal credit risk factors. Sensitive attributes present in the feature set require governance review before any deployment in a regulated lending context under ECOA or equivalent frameworks.
