"""Shared pytest fixtures backed by the canonical project data directory."""

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "processed"
CLEAN_DIR = DATA_DIR / "clean"
LEAKY_DIR = DATA_DIR / "leaky"
TARGET_COL = "Churn"
ID_COL = "customerID"


def _load(split_dir: str, name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / split_dir / name)


@pytest.fixture
def clean_train():
    return _load("clean", "clean_train.csv")


@pytest.fixture
def clean_test():
    return _load("clean", "clean_test.csv")


@pytest.fixture
def leaky_overlap_train():
    return _load("leaky", "leaky_overlap_train.csv")


@pytest.fixture
def leaky_overlap_test():
    return _load("leaky", "leaky_overlap_test.csv")


@pytest.fixture
def leaky_group_overlap_train():
    return _load("leaky", "leaky_group_overlap_train.csv")


@pytest.fixture
def leaky_group_overlap_test():
    return _load("leaky", "leaky_group_overlap_test.csv")


@pytest.fixture
def clean_temporal_train():
    return _load("clean", "clean_temporal_train.csv")


@pytest.fixture
def clean_temporal_test():
    return _load("clean", "clean_temporal_test.csv")


@pytest.fixture
def leaky_temporal_feature_train():
    return _load("leaky", "leaky_temporal_feature_train.csv")


@pytest.fixture
def leaky_temporal_feature_test():
    return _load("leaky", "leaky_temporal_feature_test.csv")


@pytest.fixture
def leaky_temporal_split_train():
    return _load("leaky", "leaky_temporal_split_train.csv")


@pytest.fixture
def leaky_temporal_split_test():
    return _load("leaky", "leaky_temporal_split_test.csv")


@pytest.fixture
def leaky_target_train():
    return _load("leaky", "leaky_target_train.csv")


@pytest.fixture
def leaky_target_test():
    return _load("leaky", "leaky_target_test.csv")