"""Fairness analysis for the LightGBM credit risk model."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from src.config_loader import ensure_parent_dir, load_config
from src.model_utils import calculate_scale_pos_weight, get_feature_type_lists, load_engineered_data, load_model_artifact, make_train_test_split, save_dataframe
from src.threshold_optimizer import find_optimal_threshold
from src.train_lightgbm import build_lightgbm_pipeline


GENDER_COLUMN = "CODE_GENDER"
EDUCATION_COLUMN = "NAME_EDUCATION_TYPE"


def safe_auc(y_true, y_proba) -> float:
    """Return ROC-AUC when both classes are present."""
    if pd.Series(y_true).nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def safe_average_precision(y_true, y_proba) -> float:
    """Return Average Precision when at least one positive case is present."""
    if pd.Series(y_true).sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_proba))


def rate_metrics(y_true, y_proba, threshold: float) -> dict:
    """Calculate threshold metrics for one subgroup."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")
    return {
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "predicted_default_rate": float(y_pred.mean()),
        "actual_default_rate": float(pd.Series(y_true).mean()),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def disaggregated_metrics(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_proba: np.ndarray,
    column: str,
    threshold: float,
    allowed_groups: list[str] | None = None,
) -> pd.DataFrame:
    """Compute subgroup metrics for a sensitive or governance-relevant column."""
    rows = []
    group_values = sorted(X_test[column].dropna().unique())
    if allowed_groups is not None:
        group_values = [group for group in group_values if group in allowed_groups]
    for group_value in group_values:
        mask = X_test[column] == group_value
        subgroup_y = y_test.loc[mask]
        subgroup_proba = y_proba[mask.to_numpy()]
        metrics = rate_metrics(subgroup_y, subgroup_proba, threshold)
        rows.append(
            {
                "attribute": column,
                "group": str(group_value),
                "n": int(mask.sum()),
                "roc_auc": safe_auc(subgroup_y, subgroup_proba),
                "average_precision": safe_average_precision(subgroup_y, subgroup_proba),
                **metrics,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        gap = (
            result["false_positive_rate"].max() - result["false_positive_rate"].min()
            + result["false_negative_rate"].max()
            - result["false_negative_rate"].min()
        )
        result["equalized_odds_gap"] = float(gap)
    return result


def train_without_gender(
    full_pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict,
) -> tuple[float, float]:
    """Train a comparable LightGBM model after removing CODE_GENDER."""
    X_train_no_gender = X_train.drop(columns=[GENDER_COLUMN])
    X_test_no_gender = X_test.drop(columns=[GENDER_COLUMN])
    params = full_pipeline.named_steps["model"].get_params()
    valid_params = LGBMClassifier().get_params().keys()
    filtered_params = {key: value for key, value in params.items() if key in valid_params}
    filtered_params["scale_pos_weight"] = calculate_scale_pos_weight(y_train)
    filtered_params["random_state"] = config["model"]["random_state"]
    numeric_features, categorical_features = get_feature_type_lists(X_train_no_gender)
    no_gender_pipeline = build_lightgbm_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        params=filtered_params,
    )
    no_gender_pipeline.fit(X_train_no_gender, y_train)
    y_proba_no_gender = no_gender_pipeline.predict_proba(X_test_no_gender)[:, 1]
    no_gender_auc = roc_auc_score(y_test, y_proba_no_gender)
    full_auc = roc_auc_score(y_test, full_pipeline.predict_proba(X_test)[:, 1])
    return float(no_gender_auc), float(full_auc - no_gender_auc)


def save_fairness_visuals(gender_df: pd.DataFrame, education_df: pd.DataFrame, config: dict) -> None:
    """Save fairness charts."""
    gender_output = ensure_parent_dir(config["visuals"]["fairness_fpr_fnr_gender"])
    plot_gender = gender_df.set_index("group")[["false_positive_rate", "false_negative_rate"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_gender.plot(kind="bar", ax=ax, color=["#1B4F8A", "#B45309"])
    ax.set_title("FPR and FNR by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Rate")
    ax.legend(["FPR", "FNR"])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(gender_output, dpi=300)
    plt.close(fig)

    education_output = ensure_parent_dir(config["visuals"]["fairness_auc_education"])
    fig, ax = plt.subplots(figsize=(9, 5))
    education_df.sort_values("roc_auc").plot(
        kind="barh",
        x="group",
        y="roc_auc",
        ax=ax,
        color="#1B4F8A",
        legend=False,
    )
    ax.set_title("AUC by Education Type")
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(education_output, dpi=300)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    """Render a compact Markdown table without optional dependencies."""
    display = df[columns].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def save_fairness_report(
    gender_df: pd.DataFrame,
    education_df: pd.DataFrame,
    no_gender_auc: float,
    auc_degradation: float,
    config: dict,
) -> None:
    """Save the human-readable fairness report."""
    output_file = ensure_parent_dir(config["reports"]["fairness_report_md"])
    gender_columns = [
        "group",
        "n",
        "roc_auc",
        "average_precision",
        "false_positive_rate",
        "false_negative_rate",
        "predicted_default_rate",
        "actual_default_rate",
        "equalized_odds_gap",
    ]
    education_columns = [
        "group",
        "n",
        "roc_auc",
        "average_precision",
        "false_positive_rate",
        "false_negative_rate",
        "predicted_default_rate",
        "actual_default_rate",
        "equalized_odds_gap",
    ]
    recommendation = (
        "CODE_GENDER should be further investigated before any deployment. "
        "The current portfolio system may retain it for transparent analysis, "
        "but a regulated lender should compare policy, legal, and fairness impacts before using it operationally."
    )
    report = f"""# Fairness Report

## Gender Metrics

{dataframe_to_markdown(gender_df, gender_columns)}

## Education Metrics

{dataframe_to_markdown(education_df, education_columns)}

## AUC Impact of Removing CODE_GENDER

The LightGBM model retrained without `CODE_GENDER` achieved ROC-AUC {no_gender_auc:.4f}. The AUC degradation relative to the full model was {auc_degradation:.4f}.

## Equalized Odds in Lending

Equalized Odds compares error rates across groups. In lending, it asks whether groups experience similar false positive rates, where low-risk applicants may be unnecessarily routed to manual review, and similar false negative rates, where risky applicants may be missed by the review queue.

## Regulatory Context

Credit scoring and lending workflows require governance under the Equal Credit Opportunity Act (ECOA) in the United States, GDPR Article 22 for automated decision-making in the European Union, and the EU AI Act high-risk classification for credit scoring systems.

## Recommendation

{recommendation}
"""
    output_file.write_text(report, encoding="utf-8")


def main() -> None:
    """Run fairness analysis and save reports."""
    config = load_config()

    print("Loading LightGBM model and holdout split...")
    model = load_model_artifact(config["artifacts"]["lightgbm_model"])
    X, y = load_engineered_data()
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, config)

    y_proba = model.predict_proba(X_test)[:, 1]
    _, _, _, optimal_threshold = find_optimal_threshold(
        y_test,
        y_proba,
        fn_cost=config["thresholds"]["business_scenarios"]["lender"]["fn_cost"],
        fp_cost=config["thresholds"]["business_scenarios"]["lender"]["fp_cost"],
    )

    print("Computing subgroup metrics...")
    gender_df = disaggregated_metrics(
        X_test,
        y_test,
        y_proba,
        GENDER_COLUMN,
        optimal_threshold,
        allowed_groups=["F", "M"],
    )
    education_df = disaggregated_metrics(X_test, y_test, y_proba, EDUCATION_COLUMN, optimal_threshold)

    print("Training no-gender comparison model...")
    no_gender_auc, auc_degradation = train_without_gender(model, X_train, y_train, X_test, y_test, config)

    results = pd.concat([gender_df, education_df], ignore_index=True)
    results["optimal_threshold"] = optimal_threshold
    results["no_gender_auc"] = no_gender_auc
    results["auc_degradation_without_gender"] = auc_degradation

    print("Saving fairness reports and visuals...")
    save_dataframe(results, config["reports"]["fairness_report_csv"])
    save_fairness_report(gender_df, education_df, no_gender_auc, auc_degradation, config)
    save_fairness_visuals(gender_df, education_df, config)

    print()
    print("Fairness analysis complete.")
    print(f"Optimal threshold: {optimal_threshold:.2f}")
    print(f"No-gender AUC: {no_gender_auc:.4f}")
    print(f"AUC degradation without gender: {auc_degradation:.4f}")


if __name__ == "__main__":
    main()
