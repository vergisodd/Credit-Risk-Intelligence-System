# Calibration Report

Champion model: **LightGBM+Bureau**

## Summary

- Brier score: **0.1853**
- Holdout base default rate: **8.07%**
- Mean predicted risk score: **39.38%**

Scores are useful for prioritizing manual review. They should not be treated as perfectly calibrated probability-of-default estimates without additional validation and possible calibration.

## Calibration by Decile

|   decile |   applicant_count |   observed_default_rate |   mean_predicted_risk |   min_score |   max_score |   calibration_error |
|---------:|------------------:|------------------------:|----------------------:|------------:|------------:|--------------------:|
|   1.0000 |         6151.0000 |                  0.2869 |                0.7983 |      0.7119 |      0.9844 |              0.5114 |
|   2.0000 |         6150.0000 |                  0.1533 |                0.6523 |      0.5976 |      0.7119 |              0.4990 |
|   3.0000 |         6150.0000 |                  0.1036 |                0.5503 |      0.5067 |      0.5976 |              0.4468 |
|   4.0000 |         6151.0000 |                  0.0761 |                0.4675 |      0.4305 |      0.5066 |              0.3915 |
|   5.0000 |         6150.0000 |                  0.0610 |                0.3960 |      0.3632 |      0.4305 |              0.3350 |
|   6.0000 |         6150.0000 |                  0.0393 |                0.3321 |      0.3021 |      0.3632 |              0.2928 |
|   7.0000 |         6151.0000 |                  0.0338 |                0.2736 |      0.2451 |      0.3021 |              0.2398 |
|   8.0000 |         6150.0000 |                  0.0275 |                0.2174 |      0.1898 |      0.2451 |              0.1900 |
|   9.0000 |         6150.0000 |                  0.0166 |                0.1611 |      0.1309 |      0.1898 |              0.1445 |
|  10.0000 |         6150.0000 |                  0.0091 |                0.0896 |      0.0072 |      0.1309 |              0.0805 |

## Calibration by Risk Tier

| risk_tier   |   applicant_count |   observed_default_rate |   mean_predicted_risk |   min_score |   max_score |   calibration_error |
|:------------|------------------:|------------------------:|----------------------:|------------:|------------:|--------------------:|
| Low         |             24364 |                  0.0216 |                0.1843 |      0.0072 |      0.3000 |              0.1627 |
| Medium      |             24378 |                  0.0686 |                0.4322 |      0.3000 |      0.5900 |              0.3636 |
| High        |             12761 |                  0.2168 |                0.7206 |      0.5900 |      0.9844 |              0.5038 |

## Governance Note

Calibration needs further production validation, ideally with out-of-time data and monitored cohorts. In this portfolio project, calibration is reported as a diagnostic for score interpretation rather than as a claim that scores are legally or operationally calibrated PD estimates.
