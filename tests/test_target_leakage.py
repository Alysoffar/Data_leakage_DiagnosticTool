"""
test_target_leakage.py

Validates target_leakage.py against the clean and leaky_target fixtures.
"""

import numpy as np
import pandas as pd
import pytest

from leak_detector.target_leakage import (
    check_correlation,
    check_single_feature_predictiveness,
    is_id_like,
)
from tests.conftest import TARGET_COL


def test_clean_data_flags_nothing(clean_train):
    pred_result = check_single_feature_predictiveness(clean_train, target_col=TARGET_COL, threshold=0.9)
    corr_result = check_correlation(clean_train, target_col=TARGET_COL, threshold=0.9)
    assert pred_result["n_flagged"] == 0
    assert corr_result["n_flagged"] == 0


def test_leaky_target_flags_cheat_score(leaky_target_train):
    result = check_single_feature_predictiveness(leaky_target_train, target_col=TARGET_COL, threshold=0.9)
    flagged_columns = [f["column"] for f in result["detail"]]
    assert "cheat_score" in flagged_columns


def test_leaky_target_cheat_score_score_is_near_perfect(leaky_target_train):
    result = check_single_feature_predictiveness(leaky_target_train, target_col=TARGET_COL, threshold=0.9)
    cheat_score_flag = next(f for f in result["detail"] if f["column"] == "cheat_score")
    assert cheat_score_flag["score"] > 0.95


def test_correlation_flags_cheat_score(leaky_target_train):
    result = check_correlation(leaky_target_train, target_col=TARGET_COL, threshold=0.9)
    flagged_columns = [f["column"] for f in result["detail"]]
    assert "cheat_score" in flagged_columns


def test_id_like_column_is_not_flagged(clean_train):
    # customerID is unique per row -- it should be skipped internally
    # rather than showing up as a flagged "leaky" feature.
    result = check_single_feature_predictiveness(clean_train, target_col=TARGET_COL, threshold=0.9)
    flagged_columns = [f["column"] for f in result["detail"]]
    assert "customerID" not in flagged_columns


def test_predictiveness_requires_binary_target():
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
            "target": ["a", "b", "c", "a", "b", "c"],
        }
    )

    with pytest.raises(ValueError, match="must be binary"):
        check_single_feature_predictiveness(dataframe, target_col="target")


def test_id_heuristic_handles_differently_named_numeric_identifier():
    dataframe = pd.Series(np.arange(100), name="record_number")

    assert is_id_like(dataframe)


def test_numeric_identifier_is_skipped_by_predictiveness_check():
    dataframe = pd.DataFrame(
        {
            "record_number": np.arange(100),
            "target": [0, 1] * 50,
        }
    )

    result = check_single_feature_predictiveness(dataframe, target_col="target")

    assert "record_number" not in [flag["column"] for flag in result["detail"]]