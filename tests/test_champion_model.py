from src.champion_model import create_model_manifest, get_champion_spec, resolve_selected_threshold
from src.threshold_optimizer import find_optimal_threshold


def test_champion_spec_resolves_configured_artifact(config):
    spec = get_champion_spec(config)
    assert spec.model_name == "LightGBM+Bureau"
    assert spec.artifact_key == "lightgbm_bureau_model"
    assert spec.feature_set == "application+bureau"
    assert spec.requires_bureau is True


def test_resolve_selected_threshold_uses_named_policy(config, synthetic_y_true, synthetic_y_proba):
    result = find_optimal_threshold(synthetic_y_true, synthetic_y_proba, fn_cost=10, fp_cost=1)
    threshold = resolve_selected_threshold(config, result)
    assert threshold == result.f1_optimal_threshold


def test_model_manifest_contains_governance_fields(config, synthetic_y_true, synthetic_y_proba):
    result = find_optimal_threshold(synthetic_y_true, synthetic_y_proba, fn_cost=10, fp_cost=1)
    manifest = create_model_manifest(
        config,
        threshold_optimization=result,
        metrics={"roc_auc": 0.77, "average_precision": 0.26},
    )
    assert manifest["champion_model_name"] == "LightGBM+Bureau"
    assert (
        manifest["threshold_policy"]["cost_minimizing_threshold"]
        == result.cost_minimizing_threshold
    )
    assert manifest["threshold_policy"]["f1_optimal_threshold"] == result.f1_optimal_threshold
    assert manifest["limitations"]
