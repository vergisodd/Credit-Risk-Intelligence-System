# Business Recommendations — Credit Risk Screening System

## Executive Summary

At the lender-cost-optimal threshold of 0.50, this model captures 68.8% of future payment-difficulty cases while routing 31.7% of the applicant pool to manual review. The most important intervention is to use the score as a documented review-prioritization queue rather than an automatic approval or denial rule. The most important limitation is that default-class precision remains low, so review capacity and fairness monitoring matter as much as raw AUC.

## Operating Threshold Recommendations

| Objective | Threshold | Recall | Precision | Est. Review Volume (per 10K) | Est. Missed Defaults (per 10K) |
|---|---:|---:|---:|---:|---:|
| Maximize default capture (fn_cost=10) | 0.50 | 0.688 | 0.175 | 3,166 | 252 |
| Balanced review burden (fn_cost=3, fp_cost=3) | 0.91 | 0.026 | 0.566 | 37 | 786 |
| Precision-focused review (fp_cost=3) | 0.95 | 0.002 | 0.647 | 3 | 805 |
| F1-optimal | 0.66 | 0.444 | 0.253 | 1,415 | 449 |

## Priority Risk Segments

High Risk applicants tend to combine weak external-source risk signals with heavier repayment structures, including higher annuity burden and less favorable credit-term ratios. The SHAP analysis also shows that education category and `CODE_GENDER` influence model behavior, which means the highest-risk queue should be reviewed alongside subgroup error rates rather than treated as a purely financial ranking. In practical terms, the risk team should use the top tier to prioritize analyst attention on affordability stress and external-score weakness while documenting any sensitive-attribute governance concerns.

## Fairness Implications for Policy

The model shows a 0.2371 Equalized Odds gap between male and female applicants. Operationally, male applicants face a false positive rate of 16.7% while female applicants face 8.9%, meaning male applicants are approximately 88% more likely to be routed to manual review even when they would not default. Any threshold choice should therefore be reviewed with a fairness lens before deployment, especially if review routing creates customer friction or downstream denial risk.

## What This System Cannot Do

This system cannot make automatic credit decisions, cannot replace human judgment on edge cases, and cannot be deployed in a regulated lending workflow without legal, compliance, and model-risk review. Its appropriate role is to rank applications for manual review and to make the tradeoffs of that ranking visible to risk leadership.

## Recommended Next Investments

1. Integrate remaining relational tables to close the AUC gap.
2. Add return_reason or application outcome feedback to improve recall for specific default patterns.
3. Implement drift monitoring to track model performance across applicant cohorts over time.
