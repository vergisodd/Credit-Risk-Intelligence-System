# Explainability Report

## Global SHAP Findings

Global SHAP analysis was computed on 1,000 stratified holdout applicants using the configured champion model: **LightGBM+Bureau**. One-hot encoded features are aggregated back to source feature names where possible. The top global features are:

| Rank | Feature | Direction | Mean Abs SHAP | Interpretation |
|---:|---|---|---:|---|
| 1 | `EXT_SOURCE_MEAN` | decreases risk on average | 0.211641 | Lower external-source scores generally indicate a weaker external credit-risk signal. |
| 2 | `CREDIT_TERM_RATIO` | increases risk on average | 0.172760 | A higher annuity-to-credit relationship can indicate a shorter or heavier repayment structure. |
| 3 | `EXT_SOURCE_3` | increases risk on average | 0.162034 | This external score often captures repayment-risk signal not visible in simple affordability ratios. |
| 4 | `CODE_GENDER` | increases risk on average | 0.124259 | Gender is a sensitive demographic attribute and requires governance review before any deployment. |
| 5 | `EXT_SOURCE_MAX` | decreases risk on average | 0.123583 | The strongest external-source score can offset some weaker application attributes. |
| 6 | `GOODS_CREDIT_RATIO` | decreases risk on average | 0.117500 | This compares the financed goods price with the total requested credit. |
| 7 | `EXT_SOURCE_2` | decreases risk on average | 0.111862 | This external score adds third-party risk information beyond income and loan amount. |
| 8 | `NAME_EDUCATION_TYPE` | decreases risk on average | 0.110981 | Education category may proxy socioeconomic patterns and should be reviewed for fairness impact. |
| 9 | `AMT_ANNUITY` | increases risk on average | 0.105005 | Higher required annuity payments can increase repayment burden. |
| 10 | `AMT_GOODS_PRICE` | increases risk on average | 0.090878 | The goods price helps contextualize the requested credit amount. |

## Fairness Note

`CODE_GENDER` and `NAME_EDUCATION_TYPE` appear in the feature set and may contribute to model predictions. These variables require fairness review because they can directly encode or proxy protected and socioeconomic characteristics.

## Per-Applicant Explanation Methodology

Applicant-level explanations transform the applicant row through the fitted preprocessing pipeline, calculate SHAP values on the transformed feature vector, aggregate one-hot encoded values back to source feature names, and render a waterfall plot showing the strongest positive and negative contributors to that specific prediction. The generated reason-code table separates contributors increasing risk from contributors decreasing risk and includes raw applicant values where available.

## Limitations

Global feature importance reflects aggregate model behavior across the holdout population. SHAP values for individual applicants indicate feature contribution to that specific prediction, not causal credit risk factors and not legally sufficient adverse-action reasons. Sensitive attributes present in the feature set require governance review before any deployment in a regulated lending context under ECOA or equivalent frameworks.
