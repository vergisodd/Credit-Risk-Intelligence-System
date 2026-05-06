import numpy as np
import pytest

from src.config_loader import load_config


@pytest.fixture
def config():
    load_config.cache_clear()
    return load_config()


@pytest.fixture
def synthetic_y_true():
    y = np.array([1] * 40 + [0] * 460)
    rng = np.random.default_rng(42)
    rng.shuffle(y)
    return y


@pytest.fixture
def synthetic_y_proba():
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1, 500)
