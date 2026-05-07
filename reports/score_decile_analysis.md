# Score Decile and Lift Analysis

This report evaluates the configured champion model as a ranking tool for manual credit review. AUC is useful, but review teams also need to know whether the highest-risk score bands actually concentrate defaults.

Base default rate in this holdout split: **8.07%**.

Top score decile observed default rate: **28.69%**.

Scores are sorted descending, so decile 1 is the highest predicted-risk group.

|   decile |   applicant_count |   actual_default_count |   actual_default_rate |   mean_predicted_score |   min_score |   max_score |   lift_vs_base_rate |   cumulative_applicant_percentage |   cumulative_default_capture_rate |
|---------:|------------------:|-----------------------:|----------------------:|-----------------------:|------------:|------------:|--------------------:|----------------------------------:|----------------------------------:|
|   1.0000 |         6151.0000 |              1765.0000 |                0.2869 |                 0.7983 |      0.7119 |      0.9844 |              3.5545 |                            0.1000 |                            0.3555 |
|   2.0000 |         6150.0000 |               943.0000 |                0.1533 |                 0.6523 |      0.5976 |      0.7119 |              1.8994 |                            0.2000 |                            0.5454 |
|   3.0000 |         6150.0000 |               637.0000 |                0.1036 |                 0.5503 |      0.5067 |      0.5976 |              1.2830 |                            0.3000 |                            0.6737 |
|   4.0000 |         6151.0000 |               468.0000 |                0.0761 |                 0.4675 |      0.4305 |      0.5066 |              0.9425 |                            0.4000 |                            0.7680 |
|   5.0000 |         6150.0000 |               375.0000 |                0.0610 |                 0.3960 |      0.3632 |      0.4305 |              0.7553 |                            0.5000 |                            0.8435 |
|   6.0000 |         6150.0000 |               242.0000 |                0.0393 |                 0.3321 |      0.3021 |      0.3632 |              0.4874 |                            0.6000 |                            0.8922 |
|   7.0000 |         6151.0000 |               208.0000 |                0.0338 |                 0.2736 |      0.2451 |      0.3021 |              0.4189 |                            0.7000 |                            0.9341 |
|   8.0000 |         6150.0000 |               169.0000 |                0.0275 |                 0.2174 |      0.1898 |      0.2451 |              0.3404 |                            0.8000 |                            0.9682 |
|   9.0000 |         6150.0000 |               102.0000 |                0.0166 |                 0.1611 |      0.1309 |      0.1898 |              0.2054 |                            0.9000 |                            0.9887 |
|  10.0000 |         6150.0000 |                56.0000 |                0.0091 |                 0.0896 |      0.0072 |      0.1309 |              0.1128 |                            1.0000 |                            1.0000 |

## Interpretation

Lift above 1.0 means a score band has a higher observed default rate than the holdout base rate. This analysis supports review prioritization only; it does not approve or reject applicants.
