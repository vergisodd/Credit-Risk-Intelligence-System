# Business Impact Simulation

This report evaluates the champion model as a **manual review-routing policy**, not an automated approval or rejection engine.

Cost units use the configured lender scenario: false negatives = 10 units and false positives = 1 unit. These are relative portfolio demonstration units, not dollars and not claimed financial savings.

| policy_name               | threshold_or_capacity   |   applicants_reviewed |   review_rate |   true_defaults_captured |   default_capture_rate |   false_reviews |   false_review_rate |   missed_defaults |   precision |   recall |     f1 |   total_cost_units |
|:--------------------------|:------------------------|----------------------:|--------------:|-------------------------:|-----------------------:|----------------:|--------------------:|------------------:|------------:|---------:|-------:|-------------------:|
| Default threshold 0.50    | 0.5                     |                 18962 |        0.3083 |                     3381 |                 0.6810 |           15581 |              0.8217 |              1584 |      0.1783 |   0.6810 | 0.2826 |         31421.0000 |
| F1-optimal threshold      | 0.66                    |                  8732 |        0.1420 |                     2226 |                 0.4483 |            6506 |              0.7451 |              2739 |      0.2549 |   0.4483 | 0.3250 |         33896.0000 |
| Cost-minimizing threshold | 0.53                    |                 16709 |        0.2717 |                     3194 |                 0.6433 |           13515 |              0.8088 |              1771 |      0.1912 |   0.6433 | 0.2947 |         31225.0000 |
| Top 10% review capacity   | 10%                     |                  6151 |        0.1000 |                     1765 |                 0.3555 |            4386 |              0.7131 |              3200 |      0.2869 |   0.3555 | 0.3176 |         36386.0000 |
| Top 15% review capacity   | 15%                     |                  9226 |        0.1500 |                     2295 |                 0.4622 |            6931 |              0.7512 |              2670 |      0.2488 |   0.4622 | 0.3234 |         33631.0000 |
| Top 20% review capacity   | 20%                     |                 12301 |        0.2000 |                     2708 |                 0.5454 |            9593 |              0.7799 |              2257 |      0.2201 |   0.5454 | 0.3137 |         32163.0000 |

## Interpretation

Thresholds and review-capacity rules change the tradeoff among captured defaults, false reviews, missed defaults, precision, recall, F1, and cost units. This turns the model from a leaderboard score into a decision-support workflow that can be reviewed by analysts and stakeholders.
