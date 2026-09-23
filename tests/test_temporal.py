"""Validate temporal checks against clean and intentionally leaky fixtures."""

import pandas as pd
from leak_detector.temporal import check_feature_timestamp_order, check_split_chronology

LABEL_DATE = "label_date"
FEATURE_DATES = ["last_contact_date", "last_bill_date"]


def test_feature_timestamp_strictness_controls_same_day_dates():
    dataframe = pd.DataFrame(
        {"label": ["2024-01-02"], "feature_date": ["2024-01-02"]}
    )

    assert check_feature_timestamp_order(dataframe, "label", ["feature_date"], strict=True)[
        "n_flagged"
    ] == 1
    assert check_feature_timestamp_order(dataframe, "label", ["feature_date"], strict=False)[
        "n_flagged"
    ] == 0


def test_clean_temporal_fixture_stays_quiet(clean_temporal_train, clean_temporal_test):
    feature_result = check_feature_timestamp_order(
        clean_temporal_train, LABEL_DATE, FEATURE_DATES
    )
    split_result = check_split_chronology(clean_temporal_train, clean_temporal_test, LABEL_DATE)

    assert feature_result["n_flagged"] == 0
    assert split_result["n_flagged"] == 0
    assert split_result["detail"]["is_chronological"] is True


def test_feature_temporal_fixture_flags_feature_dates(
    leaky_temporal_feature_train, leaky_temporal_feature_test
):
    feature_result = check_feature_timestamp_order(
        leaky_temporal_feature_train, LABEL_DATE, FEATURE_DATES
    )
    split_result = check_split_chronology(
        leaky_temporal_feature_train, leaky_temporal_feature_test, LABEL_DATE
    )

    assert feature_result["n_flagged"] > 0
    assert split_result["n_flagged"] == 0


def test_split_temporal_fixture_flags_chronology(
    leaky_temporal_split_train, leaky_temporal_split_test
):
    feature_result = check_feature_timestamp_order(
        leaky_temporal_split_train, LABEL_DATE, FEATURE_DATES
    )
    split_result = check_split_chronology(
        leaky_temporal_split_train, leaky_temporal_split_test, LABEL_DATE
    )

    assert feature_result["n_flagged"] == 0
    assert split_result["n_flagged"] > 0
    assert split_result["detail"]["is_chronological"] is False