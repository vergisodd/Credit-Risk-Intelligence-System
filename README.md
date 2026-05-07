# Credit Risk Intelligence System

Credit risk review-prioritization system for the Home Credit Default Risk dataset, with a governed **LightGBM+Bureau** champion model, explicit threshold policy, SHAP reason codes, fairness diagnostics, and a Streamlit analyst console.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Analyst%20Console-red) ![License](https://img.shields.io/badge/License-MIT-green)

[Live Demo](https://credit-risk-intelligence-system.streamlit.app)  
The hosted Streamlit app is a **public demo mode** using committed reports, visuals, sample predictions, and reason codes. To use the fully interactive dashboard with custom applicant inputs and live champion-model scoring, run the project locally with the Kaggle data and trained model artifact.

![Applicant Review Console](screenshots/applicant_risk_prediction.png)


## System Workflow

```text
Raw Kaggle Data
      ↓
Cleaning + Feature Engineering
      ↓
Application + Bureau Feature Set
      ↓
Model Training + Comparison
      ↓
Champion Model Registry
      ↓
Threshold Policy + Fairness Diagnostics + SHAP
      ↓
Streamlit Analyst Console
```
---

## Champion Result

| Champion | Feature Set | ROC-AUC | Average Precision | Base Rate | Operating Threshold |
|---|---|---:|---:|---:|---:|
| LightGBM+Bureau | `application_train.csv` + `bureau.csv` aggregations | 0.7746 | 0.2681 | 8.07% | 0.66 F1-optimal |

Average Precision is reported against an 8.07% default base rate, so the lift is meaningful despite low default-class precision. This project is a manual review prioritization workflow, not an automated approval or rejection system.

## What Changed From a Generic ML Demo

- One champion model registry in `config.yaml`
- `reports/model_manifest.json` documenting artifact, feature set, metrics, threshold policy, scope, and limitations
- App, prediction, explainability, fairness, and evaluation workflows resolve the same champion model
- Thresholds are named explicitly: default, F1-optimal, cost-minimizing, and risk-tier thresholds are different concepts
- Applicant-level SHAP reason codes include positive/negative contributors and raw values where available
- Fairness diagnostics include sensitive and proxy attributes while making clear that diagnostics are not mitigation

## Analyst Console

The Streamlit app is structured like a credit risk review tool. The public cloud deployment runs in demo mode because raw Kaggle files and trained model binaries are intentionally not committed. It shows the review workflow using committed outputs, but it does not perform live model scoring.

For the actual interactive version, run locally after placing the Kaggle files in `data/raw/` and training the champion model. Local mode enables custom applicant input changes, live risk scoring, applicant-level SHAP, and threshold simulation from holdout scores.

- Champion model label and model-manifest view
- Applicant review queue sorted by risk score
- Risk tier distribution
- Threshold and review-capacity tradeoff tool
- Per-applicant SHAP waterfall and reason-code table
- Business action recommendation for manual review routing
- Governance warning when sensitive or proxy features affect an applicant explanation

Run locally:

```bash
pip install -r requirements.txt
make train-lgbm-bureau
make evaluate
make explain
make fairness
streamlit run app/streamlit_app.py
```

## Local Data Requirements

Raw Kaggle data and trained model artifacts are intentionally not committed.

Required for the full champion pipeline:

| Path | Required For |
|---|---|
| `data/raw/application_train.csv` | All training/evaluation workflows |
| `data/raw/bureau.csv` | Champion LightGBM+Bureau training, scoring, fairness, and explainability |

If `bureau.csv` is missing, the bureau champion workflow now fails clearly instead of silently pretending an application-only model is the champion.

## Reproducible Commands

```bash
pip install -r requirements.txt
make test
make train-lgbm-bureau
make evaluate
make explain
make fairness
```

`make pipeline` runs install, training, evaluation, explainability, and fairness. It requires the Kaggle files above and can take time because LightGBM tuning uses Optuna.

## Model Comparison

| Model | ROC-AUC | Average Precision | Tuned CV AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.7507 | 0.2333 | N/A |
| XGBoost | 0.7680 | 0.2575 | 0.7618 |
| LightGBM | 0.7715 | 0.2608 | 0.7655 |
| **LightGBM+Bureau** | **0.7746** | **0.2681** | **0.7702** |

See [reports/model_comparison.md](reports/model_comparison.md) and [reports/model_card.md](reports/model_card.md).

## Threshold Policy

| Threshold | Meaning |
|---|---|
| Default threshold | Conventional 0.50 classifier cutoff |
| Cost-minimizing threshold | Minimizes a stated FN/FP cost scenario |
| F1-optimal threshold | Maximizes default-class F1; configured operating threshold for this portfolio workflow |
| Risk-tier thresholds | Score bands for analyst triage, not binary decision rules |

Fairness reports no longer call the F1-optimal threshold “lender-cost-optimal.”

## Governance Notes

`CODE_GENDER`, `NAME_EDUCATION_TYPE`, and `ORGANIZATION_TYPE` are retained for transparent portfolio diagnostics. A regulated lender would need legal, compliance, and model-risk review before using sensitive or proxy attributes. Removing protected attributes does not eliminate proxy bias.

SHAP values describe model behavior, not causality, and are not legally sufficient adverse-action reasons.

## Repository Structure

```text
app/                     Streamlit analyst console
src/champion_model.py     Champion registry and manifest helpers
src/threshold_optimizer.py Named threshold optimization result
src/train_lightgbm_common.py Shared LightGBM training utilities
src/train_lightgbm_bureau.py Champion training workflow
src/evaluate_all.py       Model comparison, calibration, threshold reporting
src/explain_model.py      Champion SHAP and reason-code generation
src/fairness_analysis.py  Champion subgroup diagnostics
reports/                 Model card, manifest, fairness, explainability, comparison
tests/                   Unit tests for feature, threshold, fairness, and champion logic
```


## Key Skills Demonstrated

- Machine learning pipeline design with Logistic Regression, XGBoost, and LightGBM
- Feature engineering from relational credit bureau data
- Imbalanced classification evaluation using ROC-AUC, Average Precision, recall, precision, and F1
- Threshold optimization for review-prioritization decisions
- SHAP explainability and applicant-level reason codes
- Fairness diagnostics across sensitive and proxy attributes
- Streamlit dashboard development for analyst workflows
- Reproducible project structure with Makefile commands and unit tests


## Limitations

This is an applied ML portfolio project, not a production lending system. It lacks temporal validation, full relational-table integration, drift monitoring, adverse-action governance, and formal legal compliance review. Scores should be used only to prioritize manual review in this demonstration.

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk), used for educational and portfolio purposes.

## License

MIT License. See [LICENSE](LICENSE).
