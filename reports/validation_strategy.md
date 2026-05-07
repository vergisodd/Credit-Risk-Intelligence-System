# Validation Strategy

## Current Validation

- Uses a stratified 80/20 train/holdout split with `model.random_state`.
- Preserves the default base rate across train and holdout splits.
- Uses 5-fold cross-validation for tuned reporting where applicable.
- Reports ROC-AUC, Average Precision, threshold metrics, fairness diagnostics, explainability outputs, and model-card limitations.

## Why This Is Not Production Validation

- There is no true out-of-time validation in the current project.
- The Home Credit main application table does not provide a clean production application timestamp suitable for a reliable temporal split.
- Random splits can overestimate deployment performance when applicant populations, policies, or macro conditions shift over time.
- Current scores are review-prioritization outputs, not production probability-of-default estimates or automated lending decisions.

## Additional Checks Added

- Score decile and lift analysis to test whether high-risk score bands concentrate observed defaults.
- Calibration by score decile and risk tier to compare predicted scores with observed default rates.
- Business policy simulation using review capacity, false-negative cost units, and false-positive cost units.
- Drift monitoring simulation comparing train and holdout feature distributions.

## Recommended Production Validation

- Run out-of-time validation if a reliable time index exists.
- Track performance stability by origination cohort and product/policy segment.
- Monitor fairness and subgroup error rates over time, including sensitive and proxy attributes.
- Add post-deployment drift, calibration, and outcome monitoring before operational use.
- Compare manual review audit outcomes with model scores to test whether the workflow improves analyst prioritization.
