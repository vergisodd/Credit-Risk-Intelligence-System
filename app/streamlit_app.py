"""Interactive Streamlit dashboard for credit risk review."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config, resolve_path
from src.champion_model import (
    get_champion_spec,
    get_persisted_operating_threshold,
    load_champion_feature_data,
    review_recommendation,
)
from src.explain_model import (
    create_feature_interpretation,
    generate_applicant_reason_codes,
    generate_individual_explanation,
)
from src.feature_engineering import add_all_features
from src.feature_engineering_bureau import BUREAU_FEATURES
from src.model_utils import make_train_test_split
from src.threshold_optimizer import find_optimal_threshold


CONFIG = load_config()
CHAMPION = get_champion_spec(CONFIG)


st.set_page_config(
    page_title="Credit Risk Intelligence System",
    layout="wide",
)


def configured_path(section: str, key: str) -> Path:
    """Resolve a configured project path."""
    return resolve_path(CONFIG[section][key])


def missing_file_info(path: Path, command: str) -> None:
    """Show a consistent missing-artifact message."""
    st.info(f"Missing `{path}`. Run `{command}` to generate it.")


def load_csv(section: str, key: str, command: str) -> pd.DataFrame | None:
    """Load a configured CSV with graceful fallback."""
    path = configured_path(section, key)
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        missing_file_info(path, command)
        return None


def load_text(section: str, key: str, command: str) -> str | None:
    """Load a configured Markdown/text report with graceful fallback."""
    path = configured_path(section, key)
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        missing_file_info(path, command)
        return None


@st.cache_resource(show_spinner=False)
def load_pipeline(artifact_key: str):
    """Load a configured model artifact with graceful fallback."""
    path = configured_path("artifacts", artifact_key)
    try:
        return joblib.load(path)
    except (FileNotFoundError, ModuleNotFoundError):
        return None


@st.cache_data(show_spinner=False)
def get_holdout_scores():
    """Load holdout data and score the champion model once for dashboard tools."""
    model = load_pipeline(CHAMPION.artifact_key)
    if model is None:
        return None
    try:
        X, y = load_champion_feature_data(CONFIG)
        _, X_test, _, y_test = make_train_test_split(X, y, CONFIG)
        y_proba = model.predict_proba(X_test)[:, 1]
        return X_test, y_test, y_proba
    except FileNotFoundError:
        return None


def show_image(section: str, key: str, command: str, caption: str | None = None) -> None:
    """Display a configured image or show generation instructions."""
    path = configured_path(section, key)
    try:
        if not path.exists():
            raise FileNotFoundError(path)
        st.image(str(path), caption=caption, width="stretch")
    except FileNotFoundError:
        missing_file_info(path, command)


def style_best_values(df: pd.DataFrame):
    """Highlight best metric values in a model comparison table."""
    metric_columns = [
        column for column in df.columns
        if column != "Model" and pd.api.types.is_numeric_dtype(df[column])
    ]

    def highlight(column):
        if column.name not in metric_columns:
            return ["" for _ in column]
        best = column.max()
        return [
            "background-color: #14532D; color: #E8EDF2" if value == best else ""
            for value in column
        ]

    return df.style.apply(highlight)


def ui_risk_tier(probability: float) -> tuple[str, str, str]:
    """Return risk tier, action, and color from configured thresholds."""
    tiers = CONFIG["thresholds"]["risk_tiers"]
    if probability < tiers["low"]:
        return "LOW RISK", "Standard Processing", "#14532D"
    if probability <= tiers["medium"]:
        return "MEDIUM RISK", "Additional Review Recommended", "#B45309"
    return "HIGH RISK", "Manual Risk Review Required", "#991B1B"


def align_to_pipeline_features(pipeline, X_row: pd.DataFrame) -> pd.DataFrame:
    """Add any missing training columns before scoring a manual applicant row."""
    if hasattr(pipeline, "feature_names_in_"):
        expected_features = list(pipeline.feature_names_in_)
    else:
        expected_features = []
        for _, _, columns in pipeline.named_steps["preprocessor"].transformers_:
            expected_features.extend(list(columns))
    aligned = X_row.copy()
    for column in expected_features:
        if column not in aligned.columns:
            aligned[column] = 0 if column in BUREAU_FEATURES else np.nan
    return aligned[expected_features]


def build_manual_applicant_row(inputs: dict) -> pd.DataFrame:
    """Build and engineer one applicant row."""
    raw_row = pd.DataFrame([inputs])
    return add_all_features(raw_row)


def get_operating_threshold() -> float:
    """Use the champion-selected operating threshold when metrics are available."""
    return get_persisted_operating_threshold(CONFIG)


def source_feature_name(feature: str) -> str:
    """Map transformed feature names to known source names where possible."""
    known_features = [
        "NAME_EDUCATION_TYPE",
        "CODE_GENDER",
        "EXT_SOURCE_MEAN",
        "EXT_SOURCE_MIN",
        "EXT_SOURCE_MAX",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "CREDIT_TERM_RATIO",
        "GOODS_CREDIT_RATIO",
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "EMPLOYMENT_YEARS",
        "DAYS_EMPLOYED_CLEAN",
    ]
    for known_feature in known_features:
        if feature.startswith(known_feature):
            return known_feature
    return feature


def display_verdict(probability: float) -> None:
    """Render the applicant risk verdict banner."""
    operating_threshold = get_operating_threshold()
    tier, _, color = ui_risk_tier(probability)
    action = review_recommendation(probability, operating_threshold, CONFIG)
    st.markdown(
        f"""
        <div style="background:{color};padding:1.1rem 1.25rem;border-radius:8px;margin:0.5rem 0 1rem 0;">
            <h2 style="color:#E8EDF2;margin:0;">{tier} — {action}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_dashboard_page() -> None:
    """Risk Dashboard page."""
    st.title("Credit Risk Review Console")
    st.caption(
        f"Champion model: {CHAMPION.model_name} | Feature set: {CHAMPION.feature_set} | "
        "Manual review prioritization only"
    )
    comparison_df = load_csv("reports", "model_comparison_full", "make evaluate")
    best_auc = None
    best_ap = None
    if comparison_df is not None and not comparison_df.empty:
        best_auc = comparison_df["AUC-ROC"].max()
        best_ap = comparison_df["Average Precision"].max()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Size", "307,511")
    col2.metric("Default Rate", "8.07%")
    col3.metric("Champion / Best AUC", f"{best_auc:.4f}" if best_auc is not None else "Pending")
    col4.metric("Champion / Best AP", f"{best_ap:.4f}" if best_ap is not None else "Pending")

    scored = get_holdout_scores()
    if scored is not None:
        X_test, _, y_proba = scored
        operating_threshold = get_operating_threshold()
        queue_df = pd.DataFrame(
            {
                "applicant_row": X_test.index,
                "score": y_proba,
            }
        )
        queue_df["risk_tier"] = queue_df["score"].apply(lambda score: ui_risk_tier(score)[0].title())
        queue_df["review_recommendation"] = queue_df["score"].apply(
            lambda score: review_recommendation(score, operating_threshold, CONFIG)
        )
        queue_df = queue_df.sort_values("score", ascending=False).reset_index(drop=True)

        queue_col, tier_col = st.columns([2, 1])
        with queue_col:
            st.subheader("Applicant Review Queue")
            st.dataframe(
                queue_df.head(25).style.format({"score": "{:.2%}"}),
                width="stretch",
                hide_index=True,
            )
        with tier_col:
            st.subheader("Risk Tier Distribution")
            distribution = queue_df["risk_tier"].value_counts().rename_axis("risk_tier").reset_index(name="count")
            st.bar_chart(distribution, x="risk_tier", y="count")

    roc_col, pr_col = st.columns(2)
    with roc_col:
        show_image("visuals", "roc_comparison_all_models", "make evaluate", "ROC comparison")
    with pr_col:
        show_image("visuals", "pr_comparison_all_models", "make evaluate", "Precision-Recall comparison")

    if comparison_df is not None:
        st.dataframe(style_best_values(comparison_df), width="stretch")

    show_image("visuals", "calibration_plot", "make evaluate", "Calibration plot")


def applicant_prediction_page() -> None:
    """Applicant Risk Prediction page."""
    st.title("Applicant Review")
    st.caption(f"Scoring with champion model: {CHAMPION.model_name}")
    model = load_pipeline(CHAMPION.artifact_key)
    model_path = configured_path("artifacts", CHAMPION.artifact_key)
    if model is None:
        st.warning(f"Champion model not found at `{model_path}`. Run `make train-lgbm-bureau` first.")
        return

    education_options = [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ]
    st.sidebar.header("Applicant Inputs")
    inputs = {
        "AMT_INCOME_TOTAL": st.sidebar.number_input("Annual Income ($)", value=150000, step=5000),
        "AMT_CREDIT": st.sidebar.number_input("Credit Amount ($)", value=500000, step=10000),
        "AMT_ANNUITY": st.sidebar.number_input("Monthly Annuity ($)", value=25000, step=500),
        "AMT_GOODS_PRICE": st.sidebar.number_input("Goods Price ($)", value=450000, step=10000),
        "DAYS_BIRTH": -365.25 * st.sidebar.slider("Age (years)", 18, 75, 33),
        "DAYS_EMPLOYED": -365.25 * st.sidebar.slider("Employment Tenure (years)", 0, 45, 5),
        "CNT_FAM_MEMBERS": st.sidebar.number_input("Family Members", min_value=1, max_value=10, value=2),
        "EXT_SOURCE_1": st.sidebar.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5),
        "EXT_SOURCE_2": st.sidebar.slider("EXT_SOURCE_2", 0.0, 1.0, 0.5),
        "EXT_SOURCE_3": st.sidebar.slider("EXT_SOURCE_3", 0.0, 1.0, 0.5),
        "CODE_GENDER": st.sidebar.selectbox("Gender", ["M", "F"]),
        "NAME_EDUCATION_TYPE": st.sidebar.selectbox("Education", education_options),
    }

    if st.button("Score Applicant", type="primary"):
        engineered_row = build_manual_applicant_row(inputs)
        aligned_row = align_to_pipeline_features(model, engineered_row)
        probability = float(model.predict_proba(aligned_row)[:, 1][0])
        operating_threshold = get_operating_threshold()
        tier, _, _ = ui_risk_tier(probability)
        action = review_recommendation(probability, operating_threshold, CONFIG)
        display_verdict(probability)

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Risk Score", f"{probability:.2%}")
        metric_col2.metric("Risk Tier", tier.title())
        metric_col3.metric(
            f"Predicted Class at {operating_threshold:.2f}",
            "Review Flag" if probability >= operating_threshold else "No Review Flag",
        )
        st.write(f"Business action recommendation: **{action}**")

        st.subheader("Applicant Reason Codes")
        try:
            shap_path = configured_path("visuals", "shap_applicant_current")
            transformed_names = model.named_steps["preprocessor"].get_feature_names_out()
            transformed_names = [str(name).split("__", maxsplit=1)[-1] for name in transformed_names]
            top_features = generate_individual_explanation(model, aligned_row, transformed_names, shap_path)
            st.image(str(shap_path), width="stretch")
            reason_codes = generate_applicant_reason_codes(model, aligned_row)
            st.dataframe(reason_codes, width="stretch", hide_index=True)
            sensitive_features = {"CODE_GENDER", "NAME_EDUCATION_TYPE", "ORGANIZATION_TYPE"}
            used_sensitive = sorted(sensitive_features & set(reason_codes["feature"]))
            if used_sensitive:
                st.warning(
                    "Governance warning: sensitive or proxy attributes appear in this applicant explanation: "
                    + ", ".join(used_sensitive)
                    + ". SHAP describes model behavior, not causality or legally sufficient adverse-action reasons."
                )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            st.info(f"Applicant SHAP explanation is unavailable: {error}")

        st.subheader("Threshold Sensitivity")
        sensitivity_rows = []
        for threshold in CONFIG["thresholds"]["sensitivity_grid"]:
            flagged = probability >= threshold
            sensitivity_rows.append(
                {
                    "Threshold": threshold,
                    "Flag Status": "Flagged" if flagged else "Not Flagged",
                    "Action Recommendation": action if flagged else "Standard processing queue",
                }
            )
        st.dataframe(pd.DataFrame(sensitivity_rows), width="stretch")


def threshold_decision_page() -> None:
    """Threshold Decision Tool page."""
    st.title("Threshold and Review Capacity Tool")
    scored = get_holdout_scores()
    if scored is None:
        st.warning("Champion model outputs are unavailable. Run `make train-lgbm-bureau` and `make evaluate`.")
        return
    _, y_test, y_proba = scored

    threshold = st.slider(
        "Decision Threshold",
        min_value=float(CONFIG["thresholds"]["min"]),
        max_value=float(CONFIG["thresholds"]["max"]),
        value=float(CONFIG["thresholds"]["default"]),
        step=float(CONFIG["thresholds"]["step"]),
    )
    col_a, col_b = st.columns(2)
    fn_cost = col_a.number_input("FN Cost Weight", min_value=0.0, value=10.0)
    fp_cost = col_b.number_input("FP Cost Weight", min_value=0.0, value=1.0)
    capacity = st.slider(
        "Daily Review Capacity as % of Queue",
        min_value=1,
        max_value=50,
        value=15,
        step=1,
    )

    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    total_cost = (fn * fn_cost) + (fp * fp_cost)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Recall", f"{recall:.3f}")
    metric_cols[1].metric("Precision", f"{precision:.3f}")
    metric_cols[2].metric("F1", f"{f1:.3f}")
    metric_cols[3].metric("FPR", f"{fpr:.3f}")
    metric_cols[4].metric("FNR", f"{fnr:.3f}")
    metric_cols[5].metric("Total Cost", f"{total_cost:,.0f}")
    capacity_threshold = float(np.quantile(y_proba, 1 - capacity / 100))
    st.info(
        f"A {capacity}% review-capacity policy would review the top {int((y_proba >= capacity_threshold).sum()):,} "
        f"applicants at an implied score threshold of {capacity_threshold:.2f}."
    )

    counts_df = pd.DataFrame({"Outcome": ["TN", "FP", "FN", "TP"], "Count": [tn, fp, fn, tp]})
    st.bar_chart(counts_df, x="Outcome", y="Count")

    threshold_rows = []
    for candidate in np.round(
        np.arange(CONFIG["thresholds"]["min"], CONFIG["thresholds"]["max"] + CONFIG["thresholds"]["step"], CONFIG["thresholds"]["step"]),
        2,
    ):
        candidate_pred = (y_proba >= candidate).astype(int)
        candidate_tn, candidate_fp, candidate_fn, candidate_tp = confusion_matrix(y_test, candidate_pred).ravel()
        threshold_rows.append(
            {
                "threshold": candidate,
                "total_cost": (candidate_fn * fn_cost) + (candidate_fp * fp_cost),
                "f1": f1_score(y_test, candidate_pred, zero_division=0),
            }
        )
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_result = find_optimal_threshold(y_test, y_proba, fn_cost=fn_cost, fp_cost=fp_cost)
    f1_optimal = threshold_df.loc[threshold_df["f1"].idxmax()]
    f1_optimal_cost = float(f1_optimal["total_cost"])

    st.write(
        f"At this threshold, the model flags {int(y_pred.sum()):,} applicants for review. "
        f"Of these, {tp:,} are genuine defaults (true positives) and {fp:,} are low-risk applicants unnecessarily reviewed. "
        f"At these cost weights, this threshold costs {total_cost:,.0f} units vs. F1-optimal threshold cost of {f1_optimal_cost:,.0f} units. "
        f"The cost-minimizing threshold for these weights is {threshold_result.cost_minimizing_threshold:.2f}; "
        f"the F1-optimal threshold is {threshold_result.f1_optimal_threshold:.2f}."
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(threshold_df["threshold"], threshold_df["total_cost"], color="#1B4F8A")
    ax.axvline(threshold, color="#B45309", linestyle="--")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total Cost")
    ax.set_title("Total Cost Across Thresholds")
    ax.grid(alpha=0.25)
    st.pyplot(fig)


def fairness_page() -> None:
    """Fairness Analysis page."""
    st.title("Fairness Analysis")
    fairness_df = load_csv("reports", "fairness_report_csv", "make fairness")
    if fairness_df is not None:
        st.dataframe(fairness_df, width="stretch")
        gender_df = fairness_df[fairness_df["attribute"] == "CODE_GENDER"].copy()
        if not gender_df.empty:
            gap = gender_df["equalized_odds_gap"].max()
            styled = gender_df.style.apply(
                lambda row: [
                    "background-color: #7F1D1D; color: #E8EDF2"
                    if column in ["false_positive_rate", "false_negative_rate"] and gap > CONFIG["fairness"]["disparity_threshold"]
                    else ""
                    for column in row.index
                ],
                axis=1,
            )
            st.subheader("Gender Metrics")
            st.dataframe(styled, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        show_image("visuals", "fairness_fpr_fnr_gender", "make fairness")
    with col2:
        show_image("visuals", "fairness_auc_education", "make fairness")

    report = load_text("reports", "fairness_report_md", "make fairness")
    if report:
        st.markdown(report)


def explainability_page() -> None:
    """Explainability page."""
    st.title("Explainability")
    show_image("visuals", "shap_feature_importance", "make explain")

    col1, col2, col3 = st.columns(3)
    with col1:
        show_image("visuals", "shap_individual_low_risk", "make explain", "Low risk")
    with col2:
        show_image("visuals", "shap_individual_medium_risk", "make explain", "Medium risk")
    with col3:
        show_image("visuals", "shap_individual_high_risk", "make explain", "High risk")

    shap_df = load_csv("reports", "shap_feature_importance", "make explain")
    if shap_df is not None:
        top_df = shap_df.head(10).copy()
        top_df["interpretation"] = top_df["feature"].apply(create_feature_interpretation)
        st.dataframe(top_df, width="stretch")

    report = load_text("reports", "explainability_report", "make explain")
    if report:
        st.markdown(report)


def model_documentation_page() -> None:
    """Model Documentation page."""
    st.title("Model Documentation")
    report_specs = [
        ("model_manifest", "Champion Model Manifest"),
        ("model_card", "Model Card"),
        ("business_recommendations", "Business Recommendations"),
        ("explainability_report", "Explainability Report"),
        ("fairness_report_md", "Fairness Report"),
    ]
    for key, label in report_specs:
        path = configured_path("reports", key)
        text = load_text("reports", key, "make pipeline")
        if text:
            with st.expander(label, expanded=key == "model_card"):
                st.markdown(text)
                st.download_button(
                    f"Download {label}",
                    data=text,
                    file_name=path.name,
                    mime="text/markdown",
                )


def main() -> None:
    """Render the selected dashboard page."""
    pages = {
        "Risk Dashboard": risk_dashboard_page,
        "Applicant Risk Prediction": applicant_prediction_page,
        "Threshold Decision Tool": threshold_decision_page,
        "Fairness Analysis": fairness_page,
        "Explainability": explainability_page,
        "Model Documentation": model_documentation_page,
    }
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Page", list(pages.keys()))
    pages[selected_page]()


if __name__ == "__main__":
    main()
