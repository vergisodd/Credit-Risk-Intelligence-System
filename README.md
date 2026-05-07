# Credit Risk Intelligence System

Credit risk review-prioritization system for the Home Credit Default Risk dataset, with a governed **LightGBM+Bureau** champion model, explicit threshold policy, SHAP reason codes, fairness diagnostics, and a Streamlit analyst console.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Analyst%20Console-red) ![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/vergisodd/Credit-Risk-Intelligence-System/actions/workflows/ci.yml/badge.svg)

[Live Demo](https://credit-risk-intelligence-system.streamlit.app)  
The hosted Streamlit app is a **public demo mode** using committed reports, visuals, sample predictions, and reason codes. To use the fully interactive dashboard with custom applicant inputs and live champion-model scoring, run the project locally with the Kaggle data and trained model artifact.

![Applicant Review Console](screenshots/applicant_risk_prediction.png)


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

## Key Skills Demonstrated

- Machine learning pipeline design with Logistic Regression, XGBoost, LightGBM, and LightGBM+Bureau
- Relational feature engineering from credit bureau, previous applications, installments, POS cash, credit-card, and bureau-balance tables
- Imbalanced classification evaluation using ROC-AUC, Average Precision, recall, precision, F1, and lift
- Threshold optimization for manual review prioritization
- Business impact simulation using cost-unit and review-capacity policies
- SHAP explainability and applicant-level reason codes
- Fairness diagnostics across sensitive and proxy attributes
- Calibration and drift-monitoring reports
- Streamlit dashboard development for analyst workflows
- Reproducible workflow with Makefile, tests, Docker, Ruff/Black, and GitHub Actions CI

## System Workflow

```text
Raw Kaggle Data
      ↓
Cleaning + Feature Engineering
      ↓
Application + Bureau Champion Model
      ↓
Optional Full-Relational Research Features
      ↓
Model Training + Comparison
      ↓
Champion Model Registry
      ↓
Threshold Policy + Score Deciles + Business Simulation
      ↓
SHAP + Fairness + Calibration + Drift Reports
      ↓
Streamlit Analyst Console
```

## Analyst Console

The Streamlit app is structured like a credit risk review tool. The public cloud deployment runs in demo mode because raw Kaggle files and trained model binaries are intentionally not committed. It shows the review workflow using committed outputs, but it does not perform live model scoring.

For the actual interactive version, run locally after placing the Kaggle files in `data/raw/` and training the champion model. Local mode enables custom applicant input changes, live risk scoring, applicant-level SHAP, and threshold simulation from holdout scores.

- Champion model label and model-manifest view
- Applicant review queue sorted by risk score
- Risk tier distribution
- Threshold and review-capacity tradeoff tool
- Score decile/lift, business impact, calibration, and drift report pages
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

Optional for full-relational research training:

- `data/raw/previous_application.csv`
- `data/raw/installments_payments.csv`
- `data/raw/POS_CASH_balance.csv`
- `data/raw/credit_card_balance.csv`
- `data/raw/bureau_balance.csv`

The configured champion remains LightGBM+Bureau unless a deeper relational model is trained, reviewed, and deliberately promoted.

## Reproducible Commands

```bash
pip install -r requirements.txt
make test
make train-lgbm-bureau
make evaluate
make explain
make fairness
make score-deciles
make business-impact
make calibration
make drift
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

## Score Decile and Lift Analysis

AUC is not enough for credit review. A useful review-prioritization model should concentrate observed defaults in the highest-risk score bands.

`make score-deciles` generates `reports/score_decile_analysis.csv`, `reports/score_decile_analysis.md`, `visuals/score_decile_lift.png`, and `visuals/cumulative_default_capture.png`. The report validates whether top-risk bands have default rates above the 8.07% base rate without hard-coding results into the README.

## Business Impact Simulation

The model is evaluated as a review-routing policy, not as an automated decision engine. `make business-impact` compares default, F1-optimal, cost-minimizing, and top-capacity review policies across recall, false reviews, missed defaults, precision, F1, and relative cost units.

These are cost units for portfolio demonstration, not dollars or claimed financial savings. The goal is to connect model scores to a decision-support workflow.

## Calibration

`make calibration` reports Brier score, mean predicted risk, calibration by score decile, and calibration by risk tier. Scores are useful for ranking applicants, but they should not be treated as perfectly calibrated probability-of-default estimates without additional validation.

## Monitoring and Drift Simulation

`make drift` runs a lightweight train-vs-holdout monitoring simulation using PSI, missing-rate deltas, and summary-statistic shifts. This is a simulation of monitoring expectations, not live production monitoring.

## Docker Demo

```bash
docker build -t credit-risk-intelligence .
docker run -p 8501:8501 credit-risk-intelligence
```

Docker runs the demo-mode Streamlit app unless local Kaggle data and trained model artifacts are mounted into the container.

## Governance Notes

`CODE_GENDER`, `NAME_EDUCATION_TYPE`, and `ORGANIZATION_TYPE` are retained for transparent portfolio diagnostics. A regulated lender would need legal, compliance, and model-risk review before using sensitive or proxy attributes. Removing protected attributes does not eliminate proxy bias.

SHAP values describe model behavior, not causality, and are not legally sufficient adverse-action reasons.

Validation details are documented in [reports/validation_strategy.md](reports/validation_strategy.md).

## Repository Structure

```text
Credit-Risk-Intelligence-System/
│
├── README.md                         Project overview, results, demo, and usage guide
├── LICENSE                           MIT license
├── Makefile                          Reproducible commands for training, testing, reports, and app launch
├── requirements.txt                  Python dependencies
├── pyproject.toml                    Ruff, Black, and pytest configuration
├── config.yaml                       Central paths, artifact names, thresholds, and model settings
├── Dockerfile                        Demo-mode Streamlit container
├── .dockerignore                     Docker build exclusions
├── .gitignore                        Git exclusions for data, models, cache files, and local artifacts
├── .python-version                   Python version pin
│
├── .github/
│   └── workflows/
│       └── ci.yml                    GitHub Actions CI workflow
│
├── .devcontainer/                    Optional VS Code / GitHub Codespaces development container
│
├── .streamlit/                       Streamlit deployment/config files
│
├── app/
│   └── streamlit_app.py              Streamlit analyst console for demo and local interactive mode
│
├── src/
│   ├── config_loader.py              Config loading and path resolution
│   ├── data_cleaning.py              Application data loading, cleaning, and schema checks
│   ├── feature_engineering.py        Application-level feature engineering
│   ├── feature_engineering_bureau.py Bureau aggregation features
│   ├── feature_engineering_relational.py Optional relational-table feature engineering
│   ├── model_utils.py                Shared preprocessing, training, evaluation, plotting, and artifact helpers
│   ├── champion_model.py             Champion model registry and manifest helpers
│   ├── threshold_optimizer.py        Threshold optimization and policy logic
│   ├── train_model.py                Logistic regression baseline training
│   ├── train_xgboost.py              XGBoost training workflow
│   ├── train_lightgbm.py             Application-only LightGBM training workflow
│   ├── train_lightgbm_bureau.py      Champion LightGBM+Bureau training workflow
│   ├── train_lightgbm_full_relational.py Optional full-relational research workflow
│   ├── evaluate_all.py               Model comparison, calibration, and threshold reporting
│   ├── score_decile_analysis.py      Score decile and lift analysis
│   ├── business_impact_simulation.py Review-routing policy simulation
│   ├── calibration_report.py         Calibration diagnostics
│   ├── drift_monitoring.py           Train-vs-holdout drift simulation
│   ├── explain_model.py              SHAP explainability and reason-code generation
│   └── fairness_analysis.py          Fairness and subgroup diagnostic reporting
│
├── tests/                            Unit tests for thresholds, features, fairness, champion logic, and reports
│
├── data/
│   ├── raw/                          Local-only Kaggle raw data placeholder
│   └── processed/                    Local-only processed data placeholder
│
├── models/                           Local-only trained model artifact placeholder
│
├── reports/                          Generated model cards, manifests, metrics, diagnostics, and business reports
│
├── visuals/                          Generated charts for model performance, SHAP, fairness, calibration, drift, and business analysis
│
├── screenshots/                      App screenshots used in README/demo
│
└── notebooks/                        Exploratory notebooks with outputs stripped to reduce repo weight
```


## Limitations

This is an applied ML portfolio project, not a production lending system. It lacks true temporal validation, live production drift monitoring, adverse-action governance, and formal legal compliance review. The configured champion uses application and bureau features; deeper relational training is optional and must be rerun locally with Kaggle tables. Scores should be used only to prioritize manual review in this demonstration.

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk), used for educational and portfolio purposes.

## License

MIT License. See [LICENSE](LICENSE).
