"""
Streamlit app for the Credit Risk Intelligence System.

This app allows users to:
- View project summary
- Load the trained Logistic Regression baseline model
- Select an applicant from the dataset
- Generate default-risk predictions
- View business risk tiers
- Review model performance metrics
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


MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_baseline.joblib"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "application_train.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "baseline_model_metrics.json"
THRESHOLD_PATH = PROJECT_ROOT / "reports" / "threshold_analysis.csv"
ROC_CURVE_PATH = PROJECT_ROOT / "visuals" / "roc_curve_logistic_regression.png"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "visuals" / "confusion_matrix_threshold_0_50.png"


st.set_page_config(
    page_title="Credit Risk Intelligence System",
    page_icon="💳",
    layout="wide"
)


@st.cache_resource
def load_model():
    """
    Load trained model.
    """
    if not MODEL_PATH.exists():
        st.error("Model file not found. Run `python src/train_model.py` first.")
        st.stop()

    return joblib.load(MODEL_PATH)


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
def load_metrics():
    """
    Load saved model metrics.
    """
    if not METRICS_PATH.exists():
        return None

    with open(METRICS_PATH, "r") as file:
        return json.load(file)


@st.cache_data
def load_threshold_analysis():
    """
    Load threshold analysis table.
    """
    if not THRESHOLD_PATH.exists():
        return None

    return pd.read_csv(THRESHOLD_PATH)


def assign_risk_tier(default_probability: float) -> str:
    """
    Convert probability into business risk tier.
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
            "Model Performance",
            "Threshold Analysis"
        ]
    )

    model = load_model()
    df, X, y, applicant_ids = load_application_data()
    metrics = load_metrics()
    threshold_df = load_threshold_analysis()

    if page == "Project Overview":
        st.header("Project Overview")

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
            This project predicts default risk and translates predictions into practical
            business risk tiers: Low Risk, Medium Risk, and High Risk.
            """
        )

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

        st.dataframe(risk_tier_table, use_container_width=True)

        st.subheader("Important Note")

        st.warning(
            """
            This baseline model should not be used for automatic loan rejection.
            Its best current use case is risk screening and manual review prioritization.
            """
        )

    elif page == "Applicant Risk Prediction":
        st.header("Applicant Risk Prediction")

        st.write(
            """
            Select an applicant ID from the dataset to generate a default-risk prediction.
            """
        )

        selected_applicant_id = st.selectbox(
            "Select Applicant ID",
            applicant_ids.head(5000).tolist()
        )

        selected_index = applicant_ids[applicant_ids == selected_applicant_id].index[0]

        applicant_features = X.loc[[selected_index]]
        actual_target = int(y.loc[selected_index])

        default_probability = float(model.predict_proba(applicant_features)[:, 1][0])
        predicted_class = int(model.predict(applicant_features)[0])
        risk_tier = assign_risk_tier(default_probability)
        action = get_risk_action(risk_tier)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Default Probability", f"{default_probability:.2%}")

        with col2:
            st.metric("Risk Tier", risk_tier)

        with col3:
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
            use_container_width=True
        )

    elif page == "Model Performance":
        st.header("Model Performance")

        if metrics is None:
            st.error("Metrics file not found. Run `python src/train_model.py` first.")
            st.stop()

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")

        with col2:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

        with col3:
            st.metric(
                "Precision",
                f"{metrics['precision_default_class']:.4f}"
            )

        with col4:
            st.metric(
                "Recall",
                f"{metrics['recall_default_class']:.4f}"
            )

        with col5:
            st.metric("F1", f"{metrics['f1_default_class']:.4f}")

        st.subheader("Interpretation")

        st.write(
            """
            The model has useful ranking ability, shown by a ROC-AUC of about 0.747.
            However, precision for the default class is low, meaning many applicants
            flagged as risky are actually non-default applicants.
            """
        )

        st.write(
            """
            The baseline is better suited for screening and review prioritization
            than automatic decision-making.
            """
        )

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("ROC Curve")
            if ROC_CURVE_PATH.exists():
                st.image(str(ROC_CURVE_PATH))
            else:
                st.warning("ROC curve image not found.")

        with col_right:
            st.subheader("Confusion Matrix")
            if CONFUSION_MATRIX_PATH.exists():
                st.image(str(CONFUSION_MATRIX_PATH))
            else:
                st.warning("Confusion matrix image not found.")

    elif page == "Threshold Analysis":
        st.header("Threshold Analysis")

        if threshold_df is None:
            st.error("Threshold analysis file not found. Run `python src/evaluate_model.py` first.")
            st.stop()

        st.write(
            """
            Classification thresholds change the tradeoff between catching risky applicants
            and wrongly flagging reliable applicants.
            """
        )

        st.dataframe(threshold_df, use_container_width=True)

        st.subheader("Business Interpretation")

        st.write(
            """
            Lower thresholds catch more default cases but create more false positives.
            Higher thresholds reduce false positives but miss more risky applicants.
            """
        )

        st.write(
            """
            For this baseline model, a threshold around 0.60 or 0.70 may be more useful
            for creating a manual review queue, while 0.50 works as a broad screening baseline.
            """
        )


if __name__ == "__main__":
    main()