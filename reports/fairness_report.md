# Fairness Report

## Gender Metrics

| group | n | roc_auc | average_precision | false_positive_rate | false_negative_rate | predicted_default_rate | actual_default_rate | equalized_odds_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F | 40561 | 0.7634 | 0.2306 | 0.0889 | 0.6245 | 0.1090 | 0.0699 | 0.2371 |
| M | 20940 | 0.7734 | 0.2994 | 0.1670 | 0.4655 | 0.2044 | 0.1017 | 0.2371 |

## Education Metrics

| group | n | roc_auc | average_precision | false_positive_rate | false_negative_rate | predicted_default_rate | actual_default_rate | equalized_odds_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Academic degree | 40 | 0.8205 | 0.1250 | 0.0256 | 1.0000 | 0.0250 | 0.0250 | 0.6107 |
| Higher education | 15061 | 0.7714 | 0.1859 | 0.0532 | 0.7229 | 0.0649 | 0.0520 | 0.6107 |
| Incomplete higher | 1988 | 0.7575 | 0.2203 | 0.1283 | 0.5541 | 0.1534 | 0.0790 | 0.6107 |
| Lower secondary | 791 | 0.7369 | 0.2918 | 0.1594 | 0.5610 | 0.1884 | 0.1037 | 0.6107 |
| Secondary / secondary special | 43623 | 0.7651 | 0.2758 | 0.1358 | 0.5231 | 0.1666 | 0.0904 | 0.6107 |

## AUC Impact of Removing CODE_GENDER

The LightGBM model retrained without `CODE_GENDER` achieved ROC-AUC 0.7697. The AUC degradation relative to the full model was 0.0017.

## Equalized Odds in Lending

Equalized Odds compares error rates across groups. In lending, it asks whether groups experience similar false positive rates, where low-risk applicants may be unnecessarily routed to manual review, and similar false negative rates, where risky applicants may be missed by the review queue.

## Regulatory Context

Credit scoring and lending workflows require governance under the Equal Credit Opportunity Act (ECOA) in the United States, GDPR Article 22 for automated decision-making in the European Union, and the EU AI Act high-risk classification for credit scoring systems.

## Recommendation

CODE_GENDER should be further investigated before any deployment. The current portfolio system may retain it for transparent analysis, but a regulated lender should compare policy, legal, and fairness impacts before using it operationally.
