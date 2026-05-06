# XGBoost Explainability Report

This report summarizes global feature importance for the trained XGBoost credit risk model in the Credit Risk Intelligence System.

Explainability in this project is used to understand broad model behavior and support business interpretation. It is not a substitute for credit policy, fairness review, or human judgment.

## Method

- Data source: `application_train.csv`
- Model: trained XGBoost sklearn Pipeline from `models/xgboost_credit_risk_model.joblib`
- Preprocessing: fitted preprocessing pipeline from the trained model
- SHAP sample size: 500 holdout rows
- Feature grouping: one-hot encoded categorical features are grouped back to their original feature names

## Built-In XGBoost Importance vs. SHAP

Built-in XGBoost importance shows which transformed features the model uses heavily when building trees. It is fast and useful for a first-pass model inspection, but it can favor variables with many encoded categories.

SHAP importance estimates the average contribution size of each feature to model predictions across sampled holdout rows. In this report, SHAP values are summarized using mean absolute SHAP value and then grouped back to original feature names for readability.

## Top SHAP Features

SHAP importance is calculated as the mean absolute SHAP value across sampled holdout rows.
Higher values indicate a larger average contribution to model predictions.

| Rank | Feature | Mean Abs SHAP Value |
|---:|---|---:|
| 1 | EXT_SOURCE_MEAN | 0.425681 |
| 2 | CODE_GENDER | 0.152279 |
| 3 | CREDIT_TERM_RATIO | 0.143557 |
| 4 | GOODS_CREDIT_RATIO | 0.122029 |
| 5 | NAME_EDUCATION_TYPE | 0.115813 |
| 6 | EXT_SOURCE_3 | 0.104828 |
| 7 | FLAG_OWN_CAR | 0.085186 |
| 8 | AMT_ANNUITY | 0.074351 |
| 9 | EXT_SOURCE_MIN | 0.073594 |
| 10 | AMT_GOODS_PRICE | 0.069374 |

## Top Built-In XGBoost Features

Built-in XGBoost importance provides a fast secondary view of which features are used heavily by the model.

| Rank | Feature | XGBoost Importance |
|---:|---|---:|
| 1 | ORGANIZATION_TYPE | 0.141756 |
| 2 | EXT_SOURCE_MEAN | 0.122783 |
| 3 | OCCUPATION_TYPE | 0.057060 |
| 4 | NAME_EDUCATION_TYPE | 0.053800 |
| 5 | NAME_INCOME_TYPE | 0.043575 |
| 6 | CODE_GENDER | 0.039551 |
| 7 | EXT_SOURCE_MAX | 0.029278 |
| 8 | EXT_SOURCE_MIN | 0.029166 |
| 9 | FLAG_OWN_CAR | 0.028571 |
| 10 | NAME_FAMILY_STATUS | 0.024516 |

## Risk Driver Interpretation

In this run, the strongest SHAP drivers include `EXT_SOURCE_MEAN`, `CODE_GENDER`, `CREDIT_TERM_RATIO`, `GOODS_CREDIT_RATIO`, `NAME_EDUCATION_TYPE`. The strongest built-in XGBoost importance drivers include `ORGANIZATION_TYPE`, `EXT_SOURCE_MEAN`, `OCCUPATION_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_INCOME_TYPE`.

External source variables are important because they summarize third-party or externally derived credit-risk signals. In this dataset, these variables often provide strong risk-ranking information beyond a single income, loan amount, or demographic field.

Affordability and credit burden ratios also matter because they describe the relationship between a requested loan and the applicant's financial capacity. Features such as credit-to-income, annuity-to-income, and loan-term ratios can help identify applicants whose repayment burden may be high relative to income or loan structure.

## Outputs

- `reports/shap_feature_importance.csv`
- `reports/xgboost_feature_importance.csv`
- `reports/explainability_report.md`
- `visuals/shap_feature_importance_xgboost.png`
- `visuals/xgboost_feature_importance.png`

## Interpretation Notes

- Global feature importance explains average model behavior, not individual applicant decisions.
- Feature importance is not causal evidence.
- SHAP is computed on a sampled holdout set to keep runtime manageable.
- Categorical variables are one-hot encoded before modeling and grouped back for readability.
- The current model uses only `application_train.csv`, so it does not include bureau, previous application, or payment history tables.
- Because default-class precision remains low, this model should support manual risk review rather than automatic rejection.
- Features such as gender, family status, occupation, and education require careful governance and fairness analysis before any real lending use.
