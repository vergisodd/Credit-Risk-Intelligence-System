# Drift Monitoring Simulation

This report compares the project training split against the holdout split. It is a **simulation of monitoring expectations**, not live production monitoring.

PSI interpretation:

- PSI < 0.10: low shift
- 0.10 <= PSI < 0.25: moderate shift
- PSI >= 0.25: high shift

| feature                        |    psi | psi_interpretation   |   train_missing_rate |   holdout_missing_rate |   missing_rate_delta |   train_mean |   holdout_mean |   mean_delta |   train_median |   holdout_median |   median_delta |
|:-------------------------------|-------:|:---------------------|---------------------:|-----------------------:|---------------------:|-------------:|---------------:|-------------:|---------------:|-----------------:|---------------:|
| EXT_SOURCE_1                   | 0.0007 | low shift            |               0.5634 |                 0.5655 |               0.0022 |       0.5017 |         0.5041 |       0.0024 |         0.5054 |           0.5086 |         0.0032 |
| EXT_SOURCE_3                   | 0.0003 | low shift            |               0.1984 |                 0.1977 |              -0.0007 |       0.5109 |         0.5108 |      -0.0001 |         0.5353 |           0.5371 |         0.0018 |
| AMT_ANNUITY                    | 0.0002 | low shift            |               0.0000 |                 0.0000 |              -0.0000 |   27108.0897 |     27110.5107 |       2.4210 |     24903.0000 |       24939.0000 |        36.0000 |
| AMT_INCOME_TOTAL               | 0.0002 | low shift            |               0.0000 |                 0.0000 |               0.0000 |  168853.1807 |    168576.8773 |    -276.3034 |    147600.0000 |      146250.0000 |     -1350.0000 |
| EXT_SOURCE_MEAN                | 0.0002 | low shift            |               0.0006 |                 0.0005 |              -0.0000 |       0.5091 |         0.5098 |       0.0007 |         0.5242 |           0.5257 |         0.0014 |
| BUREAU_CLOSED_LOAN_COUNT       | 0.0001 | low shift            |               0.0000 |                 0.0000 |               0.0000 |       2.9805 |         3.0001 |       0.0196 |         2.0000 |           2.0000 |         0.0000 |
| ANNUITY_INCOME_RATIO           | 0.0001 | low shift            |               0.0000 |                 0.0000 |              -0.0000 |       0.1809 |         0.1810 |       0.0001 |         0.1628 |           0.1628 |        -0.0001 |
| EXT_SOURCE_2                   | 0.0001 | low shift            |               0.0022 |                 0.0021 |              -0.0001 |       0.5143 |         0.5147 |       0.0004 |         0.5659 |           0.5662 |         0.0002 |
| CREDIT_INCOME_RATIO            | 0.0001 | low shift            |               0.0000 |                 0.0000 |               0.0000 |       3.9594 |         3.9503 |      -0.0091 |         3.2689 |           3.2500 |        -0.0189 |
| BUREAU_AVG_DAYS_CREDIT         | 0.0001 | low shift            |               0.0000 |                 0.0000 |               0.0000 |    -928.1226 |      -927.5578 |       0.5647 |      -927.1833 |        -929.0000 |        -1.8167 |
| AMT_CREDIT                     | 0.0001 | low shift            |               0.0000 |                 0.0000 |               0.0000 |  599338.2494 |    597777.0214 |   -1561.2280 |    514777.5000 |      512064.0000 |     -2713.5000 |
| BUREAU_ACTIVE_LOAN_COUNT       | 0.0001 | low shift            |               0.0000 |                 0.0000 |               0.0000 |       1.7623 |         1.7623 |       0.0001 |         1.0000 |           1.0000 |         0.0000 |
| BUREAU_LOAN_COUNT              | 0.0001 | low shift            |               0.0000 |                 0.0000 |               0.0000 |       4.7612 |         4.7807 |       0.0195 |         4.0000 |           4.0000 |         0.0000 |
| BUREAU_AVG_DAYS_CREDIT_ENDDATE | 0.0001 | low shift            |               0.0000 |                 0.0000 |               0.0000 |     554.7198 |       549.7940 |      -4.9258 |         0.0000 |           0.0000 |         0.0000 |
| BUREAU_PROLONGED_LOAN_COUNT    | 0.0000 | low shift            |               0.0000 |                 0.0000 |               0.0000 |       0.0287 |         0.0285 |      -0.0002 |         0.0000 |           0.0000 |         0.0000 |

## Recommended Use

In production, the same checks should compare incoming application cohorts against a stable reference population, then be reviewed with performance, calibration, fairness, and manual-review outcome monitoring.
