"""Tests for the optional static preprocessing leakage checker."""

from leak_detector.pipeline_checker import check_code_leakage_extended


def test_checker_flags_fit_before_split():
    result = check_code_leakage_extended(
        "scaler.fit(X)\nX_train, X_test = train_test_split(X)\n"
    )

    assert result["n_flagged"] == 1
    assert result["detail"][0]["method"] == "fit"


def test_checker_flags_fitting_original_dataset_after_split():
    result = check_code_leakage_extended(
        "X_train, X_test = train_test_split(X)\nscaler.fit(X)\n"
    )

    assert result["n_flagged"] == 1


def test_checker_allows_pipeline_cross_validation():
    result = check_code_leakage_extended(
        "pipe = Pipeline([])\nscores = cross_val_score(pipe, X, y, cv=5)\n"
    )

    assert result["n_flagged"] == 0


def test_checker_flags_raw_cross_validation_estimator():
    result = check_code_leakage_extended(
        "scores = cross_val_score(model, X, y, cv=5)\n"
    )

    assert result["n_flagged"] == 1
    assert result["detail"][0]["method"] == "cross_val_score"


def test_checker_flags_other_cross_validation_apis():
    result = check_code_leakage_extended(
        "cross_validate(model, X, y, cv=5)\ncross_val_predict(model, X, y, cv=5)\n"
    )

    assert result["n_flagged"] == 2
    assert {item["method"] for item in result["detail"]} == {
        "cross_validate",
        "cross_val_predict",
    }


def test_checker_tracks_aliases_of_safe_split_outputs():
    result = check_code_leakage_extended(
        "X_train, X_test = train_test_split(X)\nfeatures = X_train\nscaler.fit(features)\n"
    )

    assert result["n_flagged"] == 0


def test_checker_detects_keyword_and_partial_fit_calls():
    result = check_code_leakage_extended(
        "scaler.fit(X=X)\nscaler.partial_fit(X)\n"
    )

    assert result["n_flagged"] == 2


def test_checker_flags_raw_hyperparameter_search():
    result = check_code_leakage_extended(
        "search = GridSearchCV(model, params, cv=5)\n"
    )

    assert result["n_flagged"] == 1
    assert result["detail"][0]["method"] == "GridSearchCV"


def test_checker_tracks_helper_function_arguments():
    result = check_code_leakage_extended(
        "def fit_scaler(data):\n    scaler.fit(data)\n"
        "X_train, X_test = train_test_split(X)\n"
        "fit_scaler(X_train)\n"
    )

    assert result["n_flagged"] == 0


def test_checker_flags_helper_called_with_raw_data():
    result = check_code_leakage_extended(
        "def fit_scaler(data):\n    scaler.fit(data)\n"
        "fit_scaler(X)\n"
    )

    assert result["n_flagged"] == 1


def test_checker_flags_full_dataset_fit_inside_kfold():
    result = check_code_leakage_extended(
        "for train_idx, test_idx in kfold.split(X):\n    scaler.fit(X)\n"
    )

    assert result["n_flagged"] == 1


def test_checker_reports_syntax_errors_without_raising():
    result = check_code_leakage_extended("if:")

    assert result["n_flagged"] == 0
    assert result["error"].startswith("SyntaxError:")


def test_checker_rejects_oversized_source():
    result = check_code_leakage_extended("x = 1\n" * 1_000_001)

    assert result["n_flagged"] == 0
    assert "safety limit" in result["error"]