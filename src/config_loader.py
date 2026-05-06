"""Configuration utilities for the Credit Risk Intelligence System."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE_NAME = "config.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load the project configuration file.

    Parameters
    ----------
    config_path:
        Optional override path for tests or advanced usage. When omitted, the
        loader expects the project-level configuration file.

    Returns
    -------
    dict
        Parsed YAML configuration.
    """
    path = Path(config_path) if config_path is not None else PROJECT_ROOT / CONFIG_FILE_NAME

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {path}. "
            f"Create {CONFIG_FILE_NAME} in the project root before running the pipeline."
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    config["_project_root"] = str(PROJECT_ROOT)
    return config


def resolve_path(path_value: str | Path) -> Path:
    """
    Resolve a configured project-relative path to an absolute filesystem path.
    """
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_config_path(config: dict[str, Any], section: str, key: str) -> Path:
    """
    Resolve a named path from a configuration section.
    """
    try:
        path_value = config[section][key]
    except KeyError as error:
        raise KeyError(f"Missing config path: {section}.{key}") from error
    return resolve_path(path_value)


def ensure_parent_dir(path_value: str | Path) -> Path:
    """
    Resolve a path and create its parent directory.
    """
    path = resolve_path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path_value: str | Path) -> Path:
    """
    Resolve a directory path and create it.
    """
    path = resolve_path(path_value)
    path.mkdir(parents=True, exist_ok=True)
    return path
