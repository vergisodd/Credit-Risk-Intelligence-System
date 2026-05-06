# Note: train-lgbm-bureau requires bureau.csv in data/raw/. See README for download.

.PHONY: install train-lr train-xgb train-lgbm train-lgbm-bureau train-all evaluate explain fairness app test pipeline

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

train-all: train-lr train-xgb train-lgbm train-lgbm-bureau

evaluate:
	python src/evaluate_all.py

explain:
	python src/explain_model.py

fairness:
	python src/fairness_analysis.py

app:
	streamlit run app/streamlit_app.py

test:
	pytest tests/ -v

pipeline: install train-all evaluate explain fairness
