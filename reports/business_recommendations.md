# Business Recommendations

## Executive Summary

The upgraded system is best used for manual review prioritization. LightGBM is the strongest model by holdout ROC-AUC and Average Precision, but default-class precision remains low enough that automatic decisions would be inappropriate.

## Recommended Operating Threshold by Business Objective

| Objective | Threshold | Expected Recall | Expected Precision | Review Volume per 10,000 Applicants | Missed Defaults per 10,000 Applicants |
|---|---:|---:|---:|---:|---:|
| Maximize default detection (fn_cost=10, fp_cost=1) | 0.50 | 0.6878 | 0.1754 | 3,166 | 252 |
| Balanced review (fn_cost=3, fp_cost=3) | 0.91 | 0.0258 | 0.5664 | 37 | 786 |
| Precision-focused review (fn_cost=1, fp_cost=3) | 0.95 | 0.0022 | 0.6471 | 3 | 805 |

## Interpretation

The lender-oriented scenario keeps the threshold at 0.50 because missing future defaults is much more costly than sending extra applicants to review. This produces broad review coverage and catches roughly 69% of observed defaults.

Equal-cost and precision-focused scenarios become extremely conservative because defaults are rare. These thresholds reduce review volume sharply, but they miss most default cases and are only appropriate when unnecessary review is treated as the dominant business harm.

## Recommendation

Use the LightGBM score as a triage signal. Keep final credit decisions with human reviewers, document threshold choice by business objective, and monitor subgroup error rates before any regulated deployment.
