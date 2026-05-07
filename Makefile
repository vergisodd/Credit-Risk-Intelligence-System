# Note: train-lgbm-bureau requires bureau.csv in data/raw/. See README for download.

.PHONY: install train-lr train-xgb train-lgbm train-lgbm-bureau train-lgbm-full train-all evaluate explain fairness score-deciles business-impact calibration drift relational-report app lint format format-check compile test quality pipeline

install:
	pip install -r requirements.txt

train-lr:
	python src/train_model.py

train-xgb:
	python src/train_xgboost.py

train-lgbm:
	python src/train_lightgbm.py

train-lgbm-bureau:
	python src/train_lightgbm_bureau.py

train-lgbm-full:
	python src/train_lightgbm_full_relational.py

train-all: train-lr train-xgb train-lgbm train-lgbm-bureau

evaluate:
	python src/evaluate_all.py

explain:
	python src/explain_model.py

fairness:
	python src/fairness_analysis.py

score-deciles:
	python src/score_decile_analysis.py

business-impact:
	python src/business_impact_simulation.py

calibration:
	python src/calibration_report.py

drift:
	python src/drift_monitoring.py

relational-report: train-lgbm-full

app:
	streamlit run app/streamlit_app.py

lint:
	ruff check src app tests

format:
	black src app tests

format-check:
	black --check src app tests

compile:
	python -m compileall src app tests

test:
	pytest tests/ -v

quality: lint format-check compile test

pipeline: install train-all evaluate explain fairness
