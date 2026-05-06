
"""
Streamlit app for the Credit Risk Intelligence System.

This app allows users to:
- View project summary
- Select a trained model
- Generate applicant-level credit risk predictions
- Compare Logistic Regression and XGBoost model performance
- Review threshold analysis
- Review XGBoost explainability outputs
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.append(str(SRC_PATH))

from data_cleaning import load_raw_data, prepare_features_and_target
from feature_engineering import add_domain_features


LOGISTIC_MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_baseline.joblib"
XGBOOST_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_credit_risk_model.joblib"

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "application_train.csv"

LOGISTIC_METRICS_PATH = PROJECT_ROOT / "reports" / "baseline_model_metrics.json"
XGBOOST_METRICS_PATH = PROJECT_ROOT / "reports" / "xgboost_model_metrics.json"

LOGISTIC_THRESHOLD_PATH = PROJECT_ROOT / "reports" / "threshold_analysis.csv"
XGBOOST_THRESHOLD_PATH = PROJECT_ROOT / "reports" / "xgboost_threshold_analysis.csv"

LOGISTIC_ROC_CURVE_PATH = PROJECT_ROOT / "visuals" / "roc_curve_logistic_regression.png"
XGBOOST_ROC_CURVE_PATH = PROJECT_ROOT / "visuals" / "roc_curve_xgboost.png"

LOGISTIC_CONFUSION_MATRIX_PATH = PROJECT_ROOT / "visuals" / "confusion_matrix_threshold_0_50.png"
XGBOOST_CONFUSION_MATRIX_PATH = PROJECT_ROOT / "visuals" / "confusion_matrix_xgboost_threshold_0_50.png"

SHAP_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "shap_feature_importance.csv"
XGBOOST_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "xgboost_feature_importance.csv"
EXPLAINABILITY_REPORT_PATH = PROJECT_ROOT / "reports" / "explainability_report.md"

SHAP_VISUAL_PATH = PROJECT_ROOT / "visuals" / "shap_feature_importance_xgboost.png"
XGBOOST_IMPORTANCE_VISUAL_PATH = PROJECT_ROOT / "visuals" / "xgboost_feature_importance.png"


st.set_page_config(
    page_title="Credit Risk Intelligence System",
    page_icon="💳",
    layout="wide"
)


@st.cache_resource
def load_model(model_path: str):
    """
    Load trained model from disk.
    """
    path = Path(model_path)

    if not path.exists():
        return None

    return joblib.load(path)


@st.cache_data
def load_application_data():
    """
    Load and prepare applicant data.
    """
    if not RAW_DATA_PATH.exists():
        st.error("Raw dataset not found. Place `application_train.csv` inside `data/raw/`.")
        st.stop()

    df = load_raw_data(str(RAW_DATA_PATH))

    applicant_ids = df["SK_ID_CURR"].copy()

    X, y, dropped_columns = prepare_features_and_target(df)
    X = add_domain_features(X)

    return df, X, y, applicant_ids


@st.cache_data
def load_json_file(path: str):
    """
    Load JSON file if it exists.
    """
    file_path = Path(path)

    if not file_path.exists():
        return None

    with open(file_path, "r") as file:
        return json.load(file)


@st.cache_data
def load_csv_file(path: str):
    """
    Load CSV file if it exists.
    """
    file_path = Path(path)

    if not file_path.exists():
        return None

    return pd.read_csv(file_path)


@st.cache_data
def load_text_file(path: str):
    """
    Load text file if it exists.
    """
    file_path = Path(path)

    if not file_path.exists():
        return None

    return file_path.read_text()


def assign_risk_tier(default_probability: float) -> str:
    """
    Convert predicted default probability into business risk tier.
    """
    if default_probability < 0.30:
        return "Low Risk"

    if default_probability < 0.60:
        return "Medium Risk"

    return "High Risk"


def get_risk_action(risk_tier: str) -> str:
    """
    Recommend business action based on risk tier.
    """
    if risk_tier == "Low Risk":
        return "Standard processing"

    if risk_tier == "Medium Risk":
        return "Additional review if loan amount is high"

    return "Manual risk review recommended"


def extract_metric_row(model_name: str, metrics: dict) -> dict:
    """
    Convert saved metrics JSON into a comparison row.
    """
    if metrics is None:
        return {
            "Model": model_name,
            "Accuracy": None,
            "ROC-AUC": None,
            "Precision": None,
            "Recall": None,
            "F1": None
        }

    if "default_threshold_metrics" in metrics:
        threshold_metrics = metrics["default_threshold_metrics"]

        return {
            "Model": model_name,
            "Accuracy": threshold_metrics["accuracy"],
            "ROC-AUC": metrics["roc_auc"],
            "Precision": threshold_metrics["precision_default_class"],
            "Recall": threshold_metrics["recall_default_class"],
            "F1": threshold_metrics["f1_default_class"]
        }

    return {
        "Model": model_name,
        "Accuracy": metrics["accuracy"],
        "ROC-AUC": metrics["roc_auc"],
        "Precision": metrics["precision_default_class"],
        "Recall": metrics["recall_default_class"],
        "F1": metrics["f1_default_class"]
    }


def prepare_importance_display(
    importance_df: pd.DataFrame,
    value_column: str,
    value_label: str,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Format feature importance results for dashboard display.
    """
    display_columns = [
        "rank",
        "feature",
        value_column,
        "transformed_feature_count"
    ]

    available_columns = [
        column for column in display_columns
        if column in importance_df.columns
    ]

    display_df = importance_df[available_columns].head(top_n).copy()

    display_df = display_df.rename(
        columns={
            "rank": "Rank",
            "feature": "Feature",
            value_column: value_label,
            "transformed_feature_count": "Encoded Feature Count"
        }
    )

    return display_df


def main():
    """
    Main Streamlit app.
    """
    st.title("Credit Risk Intelligence System")
    st.caption("End-to-end machine learning pipeline for loan default risk prediction")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a section",
        [
            "Project Overview",
            "Applicant Risk Prediction",
            "Model Comparison",
            "Explainability",
            "Threshold Analysis"
        ]
    )

    logistic_metrics = load_json_file(str(LOGISTIC_METRICS_PATH))
    xgboost_metrics = load_json_file(str(XGBOOST_METRICS_PATH))

    logistic_threshold_df = load_csv_file(str(LOGISTIC_THRESHOLD_PATH))
    xgboost_threshold_df = load_csv_file(str(XGBOOST_THRESHOLD_PATH))

    shap_importance_df = load_csv_file(str(SHAP_IMPORTANCE_PATH))
    xgboost_importance_df = load_csv_file(str(XGBOOST_IMPORTANCE_PATH))
    explainability_report = load_text_file(str(EXPLAINABILITY_REPORT_PATH))

    if page == "Project Overview":
        st.header("Project Overview")

        df, X, y, _ = load_application_data()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")

        with col2:
            st.metric("Raw Columns", f"{df.shape[1]:,}")

        with col3:
            st.metric("Model Features", f"{X.shape[1]:,}")

        with col4:
            default_rate = y.mean()
            st.metric("Default Rate", f"{default_rate:.2%}")

        st.subheader("Business Problem")

        st.write(
            """
            Lenders need to identify applicants who may be at higher risk of payment difficulty
            while avoiding unnecessary rejection of reliable customers.
            """
        )

        st.write(
            """
            This project predicts default risk and translates model outputs into practical
            business risk tiers: Low Risk, Medium Risk, and High Risk.
            """
        )

        st.subheader("Models Included")

        model_table = pd.DataFrame(
            {
                "Model": ["Logistic Regression", "XGBoost"],
                "Role": ["Baseline model", "Improved gradient boosting model"],
                "Purpose": [
                    "Simple benchmark with interpretable baseline performance",
                    "Stronger predictive model for risk ranking"
                ]
            }
        )

        st.dataframe(model_table, width="stretch")

        st.subheader("Risk Tier Logic")

        risk_tier_table = pd.DataFrame(
            {
                "Risk Tier": ["Low Risk", "Medium Risk", "High Risk"],
                "Default Probability Range": ["< 0.30", "0.30 to 0.59", ">= 0.60"],
                "Suggested Action": [
                    "Standard processing",
                    "Additional review if loan amount is high",
                    "Manual risk review recommended"
                ]
            }
        )

        st.dataframe(risk_tier_table, width="stretch")

        st.warning(
            """
            This system is for risk screening and manual review prioritization.
            It should not be used for automatic loan rejection.
            """
        )

    elif page == "Applicant Risk Prediction":
        st.header("Applicant Risk Prediction")

        df, X, y, applicant_ids = load_application_data()

        logistic_model = load_model(str(LOGISTIC_MODEL_PATH))
        xgboost_model = load_model(str(XGBOOST_MODEL_PATH))

        model_choice = st.selectbox(
            "Choose model",
            ["Logistic Regression", "XGBoost"]
        )

        if model_choice == "Logistic Regression":
            selected_model = logistic_model
            selected_model_path = LOGISTIC_MODEL_PATH
        else:
            selected_model = xgboost_model
            selected_model_path = XGBOOST_MODEL_PATH

        if selected_model is None:
            st.error(
                f"""
                The selected model artifact was not found:

                `{selected_model_path}`

                Train the model locally before using this page.
                """
            )
            st.stop()

        selected_applicant_id = st.selectbox(
            "Select Applicant ID",
            applicant_ids.head(5000).tolist()
        )

        selected_index = applicant_ids[applicant_ids == selected_applicant_id].index[0]

        applicant_features = X.loc[[selected_index]]
        actual_target = int(y.loc[selected_index])

        default_probability = float(selected_model.predict_proba(applicant_features)[:, 1][0])
        predicted_class = int(selected_model.predict(applicant_features)[0])
        risk_tier = assign_risk_tier(default_probability)
        action = get_risk_action(risk_tier)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Selected Model", model_choice)

        with col2:
            st.metric("Default Probability", f"{default_probability:.2%}")

        with col3:
            st.metric("Risk Tier", risk_tier)

        with col4:
            st.metric("Predicted Class", predicted_class)

        st.subheader("Recommended Business Action")
        st.info(action)

        st.subheader("Actual Historical Outcome")

        if actual_target == 1:
            st.error("Actual target: Payment difficulty")
        else:
            st.success("Actual target: No payment difficulty")

        st.subheader("Selected Applicant Snapshot")

        original_row = df.loc[[selected_index]]

        display_columns = [
            "SK_ID_CURR",
            "TARGET",
            "NAME_CONTRACT_TYPE",
            "CODE_GENDER",
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "CNT_FAM_MEMBERS"
        ]

        available_display_columns = [
            column for column in display_columns
            if column in original_row.columns
        ]

        st.dataframe(
            original_row[available_display_columns],
            width="stretch"
        )

    elif page == "Model Comparison":
        st.header("Model Comparison")

        comparison_df = pd.DataFrame(
            [
                extract_metric_row("Logistic Regression", logistic_metrics),
                extract_metric_row("XGBoost", xgboost_metrics)
            ]
        )

        st.subheader("Performance Summary")
        st.dataframe(comparison_df, width="stretch")

        if xgboost_metrics is not None and logistic_metrics is not None:
            logistic_auc = comparison_df.loc[
                comparison_df["Model"] == "Logistic Regression",
                "ROC-AUC"
            ].iloc[0]

            xgboost_auc = comparison_df.loc[
                comparison_df["Model"] == "XGBoost",
                "ROC-AUC"
            ].iloc[0]

            auc_lift = xgboost_auc - logistic_auc

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Logistic ROC-AUC", f"{logistic_auc:.4f}")

            with col2:
                st.metric("XGBoost ROC-AUC", f"{xgboost_auc:.4f}")

            with col3:
                st.metric("ROC-AUC Lift", f"{auc_lift:.4f}")

        st.subheader("Interpretation")

        st.write(
            """
            XGBoost improves ROC-AUC, accuracy, precision, recall, and F1-score compared
            with the Logistic Regression baseline. The improvement is useful but modest.
            """
        )

        st.write(
            """
            Precision for the default class remains low, so the model should be treated
            as a screening and prioritization tool, not an automatic loan rejection system.
            """
        )

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Logistic Regression ROC Curve")
            if LOGISTIC_ROC_CURVE_PATH.exists():
                st.image(str(LOGISTIC_ROC_CURVE_PATH))
            else:
                st.warning("Logistic ROC curve image not found.")

        with col_right:
            st.subheader("XGBoost ROC Curve")
            if XGBOOST_ROC_CURVE_PATH.exists():
                st.image(str(XGBOOST_ROC_CURVE_PATH))
            else:
                st.warning("XGBoost ROC curve image not found.")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Logistic Regression Confusion Matrix")
            if LOGISTIC_CONFUSION_MATRIX_PATH.exists():
                st.image(str(LOGISTIC_CONFUSION_MATRIX_PATH))
            else:
                st.warning("Logistic confusion matrix image not found.")

        with col_right:
            st.subheader("XGBoost Confusion Matrix")
            if XGBOOST_CONFUSION_MATRIX_PATH.exists():
                st.image(str(XGBOOST_CONFUSION_MATRIX_PATH))
            else:
                st.warning("XGBoost confusion matrix image not found.")

    elif page == "Explainability":
        st.header("Explainability")

        st.write(
            """
            This page summarizes which inputs most influence the trained XGBoost
            risk-ranking model. Feature importance should be used to understand
            model behavior, not to make automatic lending decisions.
            """
        )

        if shap_importance_df is None and xgboost_importance_df is None:
            st.warning(
                "Explainability outputs were not found. Run the command below after "
                "training the XGBoost model."
            )
            st.code("python src/explain_model.py")
        else:
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("SHAP Feature Importance")
                if SHAP_VISUAL_PATH.exists():
                    st.image(str(SHAP_VISUAL_PATH))
                else:
                    st.warning("SHAP feature importance visual not found.")

            with col_right:
                st.subheader("XGBoost Feature Importance")
                if XGBOOST_IMPORTANCE_VISUAL_PATH.exists():
                    st.image(str(XGBOOST_IMPORTANCE_VISUAL_PATH))
                else:
                    st.warning("XGBoost feature importance visual not found.")

            if shap_importance_df is not None:
                st.subheader("Top SHAP Risk Drivers")
                st.dataframe(
                    prepare_importance_display(
                        importance_df=shap_importance_df,
                        value_column="mean_abs_shap_value",
                        value_label="Mean Absolute SHAP Value"
                    ),
                    width="stretch"
                )

            if xgboost_importance_df is not None:
                st.subheader("Top Built-In XGBoost Features")
                st.dataframe(
                    prepare_importance_display(
                        importance_df=xgboost_importance_df,
                        value_column="xgboost_importance",
                        value_label="XGBoost Importance"
                    ),
                    width="stretch"
                )

            st.subheader("Interpretation Notes")
            st.write(
                """
                SHAP values show average contribution size across sampled holdout rows.
                Built-in XGBoost importance provides a faster secondary view based on
                how the model uses features internally.
                """
            )
            st.write(
                """
                These rankings are global explanations. They do not prove causality,
                and they do not replace applicant-level manual review.
                """
            )

            if explainability_report is not None:
                with st.expander("View Explainability Report"):
                    st.markdown(explainability_report)

    elif page == "Threshold Analysis":
        st.header("Threshold Analysis")

        model_choice = st.selectbox(
            "Choose threshold table",
            ["Logistic Regression", "XGBoost"]
        )

        if model_choice == "Logistic Regression":
            threshold_df = logistic_threshold_df
        else:
            threshold_df = xgboost_threshold_df

        if threshold_df is None:
            st.error("Threshold analysis file not found. Run the model training/evaluation scripts first.")
            st.stop()

        st.write(
            """
            Classification thresholds change the tradeoff between catching risky applicants
            and wrongly flagging reliable applicants.
            """
        )

        st.dataframe(threshold_df, width="stretch")

        st.subheader("Business Interpretation")

        st.write(
            """
            Lower thresholds catch more default cases but create more false positives.
            Higher thresholds reduce false positives but miss more risky applicants.
            """
        )

        st.write(
            """
            This threshold analysis is more useful than accuracy alone because credit risk
            is an imbalanced classification problem.
            """
        )


if __name__ == "__main__":
    main()
