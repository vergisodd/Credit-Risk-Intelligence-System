# Fairness Report

## Gender Metrics

| attribute | group | n | roc_auc | average_precision | false_positive_rate | false_negative_rate | predicted_default_rate | actual_default_rate | equalized_odds_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CODE_GENDER | F | 40561 | 0.7679 | 0.2363 | 0.0899 | 0.6199 | 0.1102 | 0.0699 | 0.2347 |
| CODE_GENDER | M | 20940 | 0.7746 | 0.3085 | 0.1655 | 0.4608 | 0.2035 | 0.1017 | 0.2347 |

## Education Metrics

| attribute | group | n | roc_auc | average_precision | false_positive_rate | false_negative_rate | predicted_default_rate | actual_default_rate | equalized_odds_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NAME_EDUCATION_TYPE | Academic degree | 40 | 0.8205 | 0.1250 | 0.0256 | 1.0000 | 0.0250 | 0.0250 | 0.6323 |
| NAME_EDUCATION_TYPE | Higher education | 15061 | 0.7734 | 0.1863 | 0.0553 | 0.7126 | 0.0673 | 0.0520 | 0.6323 |
| NAME_EDUCATION_TYPE | Incomplete higher | 1988 | 0.7688 | 0.2291 | 0.1278 | 0.5223 | 0.1554 | 0.0790 | 0.6323 |
| NAME_EDUCATION_TYPE | Lower secondary | 791 | 0.7469 | 0.3232 | 0.1580 | 0.5000 | 0.1934 | 0.1037 | 0.6323 |
| NAME_EDUCATION_TYPE | Secondary / secondary special | 43623 | 0.7684 | 0.2840 | 0.1353 | 0.5218 | 0.1663 | 0.0904 | 0.6323 |

## Organization-Type Proxy Diagnostics

| attribute | group | n | roc_auc | average_precision | false_positive_rate | false_negative_rate | predicted_default_rate | actual_default_rate | equalized_odds_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ORGANIZATION_TYPE | Advertising | 98 | 0.8062 | 0.2693 | 0.0652 | 0.5000 | 0.0918 | 0.0612 | 1.3571 |
| ORGANIZATION_TYPE | Agriculture | 479 | 0.6963 | 0.2879 | 0.1563 | 0.5227 | 0.1858 | 0.0919 | 1.3571 |
| ORGANIZATION_TYPE | Bank | 507 | 0.7271 | 0.1558 | 0.0698 | 0.7500 | 0.0769 | 0.0394 | 1.3571 |
| ORGANIZATION_TYPE | Business Entity Type 1 | 1164 | 0.7419 | 0.2706 | 0.1239 | 0.5758 | 0.1495 | 0.0851 | 1.3571 |
| ORGANIZATION_TYPE | Business Entity Type 2 | 2122 | 0.7937 | 0.3074 | 0.1287 | 0.5116 | 0.1579 | 0.0811 | 1.3571 |
| ORGANIZATION_TYPE | Business Entity Type 3 | 13556 | 0.7775 | 0.3004 | 0.1482 | 0.4910 | 0.1821 | 0.0941 | 1.3571 |
| ORGANIZATION_TYPE | Cleaning | 48 | 0.9349 | 0.6969 | 0.2558 | 0.0000 | 0.3333 | 0.1042 | 1.3571 |
| ORGANIZATION_TYPE | Construction | 1322 | 0.7489 | 0.2724 | 0.2090 | 0.4357 | 0.2466 | 0.1059 | 1.3571 |
| ORGANIZATION_TYPE | Culture | 83 | 0.6083 | 0.0706 | 0.0625 | 1.0000 | 0.0602 | 0.0361 | 1.3571 |
| ORGANIZATION_TYPE | Electricity | 193 | 0.8095 | 0.2733 | 0.0968 | 0.4286 | 0.1140 | 0.0363 | 1.3571 |
| ORGANIZATION_TYPE | Emergency | 102 | 0.6729 | 0.4919 | 0.1064 | 0.5000 | 0.1373 | 0.0784 | 1.3571 |
| ORGANIZATION_TYPE | Government | 2063 | 0.7297 | 0.2001 | 0.1141 | 0.6405 | 0.1323 | 0.0742 | 1.3571 |
| ORGANIZATION_TYPE | Hotel | 183 | 0.7417 | 0.2618 | 0.0760 | 0.6667 | 0.0929 | 0.0656 | 1.3571 |
| ORGANIZATION_TYPE | Housing | 587 | 0.8469 | 0.4087 | 0.1132 | 0.4167 | 0.1516 | 0.0818 | 1.3571 |
| ORGANIZATION_TYPE | Industry: type 1 | 193 | 0.7699 | 0.2831 | 0.2023 | 0.4500 | 0.2383 | 0.1036 | 1.3571 |

## Threshold Policy

Metrics in this report use the configured champion operating threshold: **f1_optimal_threshold = 0.66** for **LightGBM+Bureau**. This is not labelled as lender-cost-optimal unless it is the `cost_minimizing_threshold`.

## AUC Impact of Removing CODE_GENDER

The champion model retrained without `CODE_GENDER` achieved ROC-AUC 0.7737. The AUC degradation relative to the full model was 0.0010.

## Equalized Odds in Lending

Equalized Odds compares error rates across groups. In lending, it asks whether groups experience similar false positive rates, where low-risk applicants may be unnecessarily routed to manual review, and similar false negative rates, where risky applicants may be missed by the review queue.

## Regulatory Context

Credit scoring and lending workflows require governance under the Equal Credit Opportunity Act (ECOA) in the United States, GDPR Article 22 for automated decision-making in the European Union, and the EU AI Act high-risk classification for credit scoring systems. Retaining sensitive variables in this portfolio experiment is for transparency and diagnostic review. Removing protected attributes does not eliminate proxy bias, and these metrics are diagnostic rather than a mitigation strategy.

## Recommendation

CODE_GENDER should be further investigated before any deployment. The current portfolio system may retain it for transparent analysis, but a regulated lender should compare policy, legal, and fairness impacts before using it operationally.
