# Explainability Report

## Global Feature Importance

Global SHAP analysis shows that the strongest model drivers are external credit-risk signals, repayment burden, and governance-sensitive applicant attributes. `EXT_SOURCE_MEAN` is the largest feature by mean absolute SHAP value, followed by `CREDIT_TERM_RATIO`, `CODE_GENDER`, `EXT_SOURCE_3`, and `GOODS_CREDIT_RATIO`. In credit risk terms, the model is primarily ranking applicants by external risk scores and affordability structure, while also using demographic and education variables that require governance review. These patterns support the model's use as a review-prioritization tool, not as a standalone automated decision engine.

## Top 10 Features

| Rank | Feature | Mean \|SHAP\| | Direction | Business Interpretation |
|---:|---|---:|---|---|
| 1 | `EXT_SOURCE_MEAN` | 0.380033 | Decreases risk | Lower external-source scores generally indicate a weaker external credit-risk signal. |
| 2 | `CREDIT_TERM_RATIO` | 0.185655 | Increases risk | A higher annuity-to-credit relationship can indicate a shorter or heavier repayment structure. |
| 3 | `CODE_GENDER` | 0.134418 | Increases risk | Gender is a sensitive demographic attribute and requires governance review before any deployment. |
| 4 | `EXT_SOURCE_3` | 0.132076 | Decreases risk | This external score often captures repayment-risk signal not visible in simple affordability ratios. |
| 5 | `GOODS_CREDIT_RATIO` | 0.116276 | Decreases risk | This compares the financed goods price with the total requested credit. |
| 6 | `NAME_EDUCATION_TYPE` | 0.108848 | Increases risk | Education category may proxy socioeconomic patterns and should be reviewed for fairness impact. |
| 7 | `EXT_SOURCE_MAX` | 0.092381 | Increases risk | The strongest external-source score can offset some weaker application attributes. |
| 8 | `FLAG_OWN_CAR` | 0.090737 | Decreases risk | This feature contributes predictive signal in the model but should not be interpreted as causal evidence. |
| 9 | `AMT_ANNUITY` | 0.090394 | Increases risk | Higher required annuity payments can increase repayment burden. |
| 10 | `AMT_GOODS_PRICE` | 0.085415 | Increases risk | The goods price helps contextualize the requested credit amount. |

## Individual Applicant Explanations

The system generates SHAP waterfall charts for individual applicants showing which features pushed the predicted probability above or below the population mean. Three representative examples are stored in visuals/:

- **Low Risk** (`shap_individual_low_risk.png`): predicted probability below 0.20
- **Medium Risk** (`shap_individual_medium_risk.png`): predicted probability 0.35–0.50
- **High Risk** (`shap_individual_high_risk.png`): predicted probability above 0.70

In each chart, red bars indicate features that increased the predicted default probability above the base value. Blue bars indicate features that decreased it. The bar length represents the magnitude of contribution.

## Limitations

SHAP values describe how the model uses each feature to generate a prediction. They do not establish causal relationships between features and credit default. A high SHAP contribution from ANNUITY_INCOME_RATIO means the model weighted that feature heavily for this applicant — it does not mean annuity burden caused the applicant's financial difficulty.

Sensitive attributes present in the feature set (`CODE_GENDER`, `NAME_EDUCATION_TYPE`) appear in SHAP output because the model uses them. Their presence in global feature importance rankings requires governance review before any regulated deployment. See `reports/fairness_report.md` for disaggregated analysis.

Global feature importance reflects aggregate model behavior across the holdout population and does not constitute a causal explanation for individual credit decisions. SHAP values for individual applicants indicate feature contribution to that specific prediction. Sensitive attributes present in the feature set require governance review before any deployment in a regulated lending context under ECOA or equivalent frameworks.
