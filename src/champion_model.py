"""Champion model registry helpers.

This module is intentionally small: it gives every scoring workflow one
authoritative place to resolve the champion artifact, feature set, and
threshold policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config_loader import ensure_parent_dir, load_config, resolve_path
from src.model_utils import (
    load_engineered_data,
    load_engineered_data_with_bureau,
    load_model_artifact,
    make_train_test_split,
    save_json,
)
from src.threshold_optimizer import ThresholdOptimizationResult


@dataclass(frozen=True)
class ChampionModelSpec:
    """Resolved champion model registry entry."""

    model_name: str
    artifact_key: str
    metrics_key: str
    threshold_analysis_key: str
    feature_set: str
    requires_bureau: bool
    fallback_artifact_key: str | None
    selected_operating_threshold: str
    artifact_path: Path
    metrics_path: Path


def get_champion_spec(config: dict[str, Any] | None = None) -> ChampionModelSpec:
    """Return the configured champion model with resolved artifact paths."""
    config = config or load_config()
    champion = config.get("champion", {})
    required = [
        "model_name",
        "artifact_key",
        "metrics_key",
        "threshold_analysis_key",
        "feature_set",
        "selected_operating_threshold",
    ]
    missing = [key for key in required if key not in champion]
    if missing:
        raise KeyError(f"Missing champion config keys: {', '.join(missing)}")

    artifact_key = champion["artifact_key"]
    metrics_key = champion["metrics_key"]
    if artifact_key not in config["artifacts"]:
        raise KeyError(f"Champion artifact_key {artifact_key!r} is not defined in artifacts.")
    if metrics_key not in config["reports"]:
        raise KeyError(f"Champion metrics_key {metrics_key!r} is not defined in reports.")

    return ChampionModelSpec(
        model_name=str(champion["model_name"]),
        artifact_key=str(artifact_key),
        metrics_key=str(metrics_key),
        threshold_analysis_key=str(champion["threshold_analysis_key"]),
        feature_set=str(champion["feature_set"]),
        requires_bureau=bool(champion.get("requires_bureau", False)),
        fallback_artifact_key=champion.get("fallback_artifact_key"),
        selected_operating_threshold=str(champion["selected_operating_threshold"]),
        artifact_path=resolve_path(config["artifacts"][artifact_key]),
        metrics_path=resolve_path(config["reports"][metrics_key]),
    )


def load_champion_model(config: dict[str, Any] | None = None) -> object:
    """Load the configured champion model artifact with a registry-aware error."""
    config = config or load_config()
    spec = get_champion_spec(config)
    if not spec.artifact_path.exists():
        raise FileNotFoundError(
            f"Champion model artifact not found at {spec.artifact_path}. "
            f"Run `make train-lgbm-bureau` to train {spec.model_name}."
        )
    return load_model_artifact(spec.artifact_path)


def load_champion_feature_data(
    config: dict[str, Any] | None = None,
    validate_schema: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the feature matrix that matches the champion model."""
    config = config or load_config()
    spec = get_champion_spec(config)
    if spec.feature_set == "application+bureau":
        return load_engineered_data_with_bureau(
            validate_schema=validate_schema,
            allow_missing_bureau=False,
        )
    if spec.feature_set == "application":
        return load_engineered_data(validate_schema=validate_schema)
    raise ValueError(f"Unsupported champion feature_set: {spec.feature_set}")


def get_champion_holdout(
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return the standard train/test split for the champion feature set."""
    config = config or load_config()
    X, y = load_champion_feature_data(config)
    return make_train_test_split(X, y, config)


def load_champion_metrics(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load champion metrics JSON if available, otherwise return an empty dict."""
    spec = get_champion_spec(config)
    if not spec.metrics_path.exists():
        return {}
    with spec.metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model_manifest(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the persisted champion manifest if available."""
    config = config or load_config()
    manifest_path = resolve_path(config["reports"]["model_manifest"])
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_persisted_operating_threshold(config: dict[str, Any] | None = None) -> float:
    """Read selected operating threshold from metrics or manifest, then fallback to default."""
    config = config or load_config()
    selected_policy = config["champion"]["selected_operating_threshold"]
    metrics_policy = load_champion_metrics(config).get("threshold_policy", {})
    if selected_policy in metrics_policy:
        return float(metrics_policy[selected_policy])

    manifest_policy = load_model_manifest(config).get("threshold_policy", {})
    if "selected_operating_threshold" in manifest_policy:
        return float(manifest_policy["selected_operating_threshold"])
    if selected_policy in manifest_policy:
        return float(manifest_policy[selected_policy])
    return float(config["thresholds"]["default"])


def resolve_selected_threshold(
    config: dict[str, Any],
    optimization: ThresholdOptimizationResult | None = None,
) -> float:
    """Resolve the configured operating threshold from an explicit policy name."""
    spec = get_champion_spec(config)
    policy = spec.selected_operating_threshold
    if policy == "default_threshold":
        return float(config["thresholds"]["default"])
    if optimization is None:
        raise ValueError(f"Threshold policy {policy!r} requires threshold optimization results.")
    if not hasattr(optimization, policy):
        raise ValueError(f"Unsupported selected operating threshold policy: {policy}")
    return float(getattr(optimization, policy))


def risk_tier(default_probability: float, config: dict[str, Any] | None = None) -> str:
    """Convert a score into the configured risk tier name."""
    config = config or load_config()
    tiers = config["thresholds"]["risk_tiers"]
    if default_probability < tiers["low"]:
        return "Low"
    if default_probability < tiers["medium"]:
        return "Medium"
    return "High"


def review_recommendation(default_probability: float, operating_threshold: float, config: dict[str, Any] | None = None) -> str:
    """Return a business-readable manual review recommendation."""
    tier = risk_tier(default_probability, config)
    if default_probability >= operating_threshold:
        return f"Route to manual review ({tier.lower()} risk tier)"
    if tier == "High":
        return "Analyst review recommended due to high risk tier"
    return "Standard processing queue; no automated approval decision"


def create_model_manifest(
    config: dict[str, Any],
    threshold_optimization: ThresholdOptimizationResult | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the champion model manifest for governance review."""
    spec = get_champion_spec(config)
    metrics = metrics if metrics is not None else load_champion_metrics(config)
    selected_threshold = (
        resolve_selected_threshold(config, threshold_optimization)
        if threshold_optimization is not None
        else None
    )
    threshold_policy = {
        "default_threshold": float(config["thresholds"]["default"]),
        "selected_operating_threshold_policy": spec.selected_operating_threshold,
        "selected_operating_threshold": selected_threshold,
        "risk_tiers": config["thresholds"]["risk_tiers"],
    }
    if threshold_optimization is not None:
        threshold_policy.update(threshold_optimization.to_summary_dict())

    return {
        "champion_model_name": spec.model_name,
        "artifact_key": spec.artifact_key,
        "artifact_path": str(spec.artifact_path.relative_to(resolve_path("."))),
        "feature_set": spec.feature_set,
        "requires_bureau": spec.requires_bureau,
        "fallback_artifact_key": spec.fallback_artifact_key,
        "metrics": {
            "roc_auc": metrics.get("roc_auc"),
            "average_precision": metrics.get("average_precision"),
            "tuned_cv_auc_mean": metrics.get("tuned_cv_auc_mean"),
            "tuned_cv_auc_std": metrics.get("tuned_cv_auc_std"),
            "tuned_cv_ap_mean": metrics.get("tuned_cv_ap_mean"),
            "default_threshold_metrics": metrics.get("default_threshold_metrics"),
        },
        "threshold_policy": threshold_policy,
        "training_data_scope": config["champion"].get("training_data_scope", {}),
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "version_info": {
            "model_random_state": config["model"]["random_state"],
            "test_size": config["model"]["test_size"],
            "cv_folds": config["model"]["cv_folds"],
        },
        "limitations": config["champion"].get("limitations", []),
    }


def save_model_manifest(
    config: dict[str, Any],
    threshold_optimization: ThresholdOptimizationResult | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist reports/model_manifest.json and return its content."""
    manifest = create_model_manifest(config, threshold_optimization, metrics)
    output_path = ensure_parent_dir(config["reports"]["model_manifest"])
    save_json(manifest, output_path)
    return manifest
