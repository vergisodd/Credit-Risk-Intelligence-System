from pathlib import Path

import pytest

from src.config_loader import load_config


def test_load_config_returns_dict():
    load_config.cache_clear()
    config = load_config()
    assert isinstance(config, dict)


def test_load_config_required_top_level_keys_exist():
    load_config.cache_clear()
    config = load_config()
    for key in ["paths", "model", "lightgbm", "thresholds", "shap"]:
        assert key in config


def test_load_config_missing_file_raises_file_not_found(tmp_path: Path):
    load_config.cache_clear()
    missing_path = tmp_path / "missing_config.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing_path)
