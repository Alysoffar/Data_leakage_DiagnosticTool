"""Validate overlap checks against clean, exact-leak, and group-leak data."""

import pandas as pd
from leak_detector.overlap import check_exact_duplicates, check_group_overlap, check_near_duplicates

from tests.conftest import ID_COL

# On real-world categorical data, a small number of coincidental exact
# matches is normal once an ID column is excluded -- this is NOT a bug.
# The bound here is generous on purpose; what actually matters is the
# comparative test below (clean must be far lower than leaky).
CLEAN_BASELINE_MAX = 20


def test_exact_duplicates_clean_data_stays_low(clean_train, clean_test):
    result = check_exact_duplicates(clean_train, clean_test, ignore_cols=[ID_COL])
    assert result["n_flagged"] <= CLEAN_BASELINE_MAX


def test_exact_duplicates_leaky_data_flagged(leaky_overlap_train, leaky_overlap_test):
    result = check_exact_duplicates(leaky_overlap_train, leaky_overlap_test, ignore_cols=[ID_COL])
    # 5% of the test set was deliberately copied into training
    assert result["n_flagged"] > 0


def test_leaky_overlap_far_exceeds_clean_baseline(
    clean_train, clean_test, leaky_overlap_train, leaky_overlap_test
):
    clean_result = check_exact_duplicates(clean_train, clean_test, ignore_cols=[ID_COL])
    leaky_result = check_exact_duplicates(leaky_overlap_train, leaky_overlap_test, ignore_cols=[ID_COL])
    # the whole point of the injected leak is that it dwarfs coincidental overlap
    assert leaky_result["n_flagged"] > clean_result["n_flagged"] * 5


def test_near_duplicates_returns_valid_shape(clean_train, clean_test):
    result = check_near_duplicates(clean_train, clean_test, threshold=0.95)
    assert result["check"] == "near_duplicate"
    assert isinstance(result["n_flagged"], int)
    assert isinstance(result["detail"], list)


def test_near_duplicates_can_limit_sample_size(clean_train, clean_test):
    result = check_near_duplicates(
        clean_train,
        clean_test,
        ignore_cols=[ID_COL],
        target_col="Churn",
        max_rows=25,
    )

    assert result["check"] == "near_duplicate"
    assert isinstance(result["n_flagged"], int)


def test_overlap_checks_handle_empty_test_set(clean_train):
    empty_test = clean_train.iloc[0:0]
    exact_result = check_exact_duplicates(clean_train, empty_test, ignore_cols=[ID_COL])
    near_result = check_near_duplicates(clean_train, empty_test, threshold=0.95)
    assert exact_result["n_flagged"] == 0
    assert near_result["n_flagged"] == 0


def test_group_overlap_flags_shared_ids_with_changed_values():
    train = pd.DataFrame(
        {"customerID": ["a", "b"], "visit_value": [1, 2]}
    )
    test = pd.DataFrame(
        {"customerID": ["b", "c"], "visit_value": [3, 4]}
    )

    result = check_group_overlap(train, test, "customerID")

    assert result["n_flagged"] == 1
    assert result["detail"] == [{"group": "b", "train_count": 1, "test_count": 1}]


def test_group_overlap_clean_fixture_stays_at_zero(clean_train, clean_test):
    result = check_group_overlap(clean_train, clean_test, ID_COL)

    assert result["n_flagged"] == 0
    assert result["detail"] == []


def test_group_overlap_leaky_fixture_is_flagged(leaky_group_overlap_train, leaky_group_overlap_test):
    result = check_group_overlap(leaky_group_overlap_train, leaky_group_overlap_test, ID_COL)

    assert result["n_flagged"] > 0
    assert len(result["detail"]) > 0


def test_group_overlap_leaky_fixture_exceeds_clean_fixture(
    clean_train, clean_test, leaky_group_overlap_train, leaky_group_overlap_test
):
    clean_result = check_group_overlap(clean_train, clean_test, ID_COL)
    leaky_result = check_group_overlap(leaky_group_overlap_train, leaky_group_overlap_test, ID_COL)

    assert leaky_result["n_flagged"] > clean_result["n_flagged"]