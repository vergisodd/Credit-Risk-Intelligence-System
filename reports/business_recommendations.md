# Business Recommendations - Credit Risk Screening System

## Executive Summary

Use the champion **LightGBM+Bureau** model as a manual-review prioritization tool, not an automatic lending decision engine. The model ranks applicants materially better than the base-rate benchmark, but default-class precision remains low because defaults are rare. Review capacity, threshold selection, calibration, and fairness monitoring matter as much as ROC-AUC.

## Operating Threshold Recommendations

| Policy Concept | Threshold | Purpose | Operational Meaning |
|---|---:|---|---|
| Default classifier threshold | 0.50 | Conventional ML cutoff | Useful for benchmark metrics; not automatically the best business policy |
| Lender cost-minimizing threshold | 0.53 | Minimize FN cost weighted 10x FP cost | Captures more defaults but routes a large queue to review |
| F1-optimal threshold | 0.66 | Balance default precision and recall | Configured operating threshold for portfolio review prioritization |
| Risk-tier thresholds | 0.30 / 0.59 | Analyst triage bands | Low/medium/high queue labels, not binary decision rules |

At threshold 0.50, the bundled champion metrics show recall of 68.1% and precision of 17.8%. That means the model is valuable for ranking and triage, but every flagged applicant still needs human review and policy context.

## Review Queue Policy

Risk teams should sort applicants by score descending, then apply a capacity-aware review policy. If analyst capacity is constrained, review the highest-scoring applicants first rather than treating all scores above a static threshold as equally urgent.

The Streamlit app now exposes both a threshold slider and a review-capacity tool so stakeholders can see the tradeoff among review volume, missed defaults, precision, recall, false positives, and false negatives.

## Fairness and Governance

`CODE_GENDER`, `NAME_EDUCATION_TYPE`, and `ORGANIZATION_TYPE` are retained in this portfolio experiment so their influence can be measured transparently. A regulated lender would need legal, compliance, and model-risk approval before using sensitive or proxy attributes. Removing protected attributes does not eliminate proxy bias, so subgroup diagnostics should continue even after feature changes.

## What This System Cannot Do

This system cannot approve or reject loans, cannot replace human judgment, and cannot be considered production-ready. Its appropriate role is to demonstrate a governed analytics workflow for manual review prioritization.

## Recommended Next Investments

1. Integrate the remaining Home Credit relational tables for stronger credit-history signal.
2. Add probability calibration and calibration monitoring before treating scores as probability estimates.
3. Add drift monitoring and periodic subgroup performance review.
4. Document any future fairness mitigation separately from diagnostic reporting.
