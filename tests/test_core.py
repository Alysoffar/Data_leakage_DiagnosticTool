"""
test_core.py

End-to-end test: runs the whole pipeline through run_leak_check() and
checks the summary dict, exactly like a real user of the tool would.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from leak_detector.core import InputDataError, main, run_leak_check

from tests.conftest import CLEAN_DIR, ID_COL, LEAKY_DIR, TARGET_COL


def test_clean_pair_reports_low_leakage(tmp_path):
    output_path = tmp_path / "clean_report.md"
    summary = run_leak_check(
        train_path=CLEAN_DIR / "clean_train.csv",
        test_path=CLEAN_DIR / "clean_test.csv",
        target_col=TARGET_COL,
        output_path=str(output_path),
        ignore_cols=[ID_COL],
    )
    assert summary["target_predictiveness_flagged"] == 0
    assert summary["target_correlation_flagged"] == 0
    assert output_path.exists()


def test_leaky_overlap_pair_is_flagged(tmp_path):
    output_path = tmp_path / "leaky_overlap_report.md"
    summary = run_leak_check(
        train_path=LEAKY_DIR / "leaky_overlap_train.csv",
        test_path=LEAKY_DIR / "leaky_overlap_test.csv",
        target_col=TARGET_COL,
        output_path=str(output_path),
        ignore_cols=[ID_COL],
    )
    assert summary["exact_overlap_flagged"] > 0
    assert summary["target_predictiveness_flagged"] == 0


def test_leaky_target_pair_is_flagged(tmp_path):
    output_path = tmp_path / "leaky_target_report.md"
    summary = run_leak_check(
        train_path=LEAKY_DIR / "leaky_target_train.csv",
        test_path=LEAKY_DIR / "leaky_target_test.csv",
        target_col=TARGET_COL,
        output_path=str(output_path),
        ignore_cols=[ID_COL],
    )
    assert summary["target_predictiveness_flagged"] > 0
    assert summary["exact_overlap_flagged"] == 0


def test_group_overlap_is_reported_end_to_end(tmp_path):
    train = pd.read_csv(CLEAN_DIR / "clean_train.csv")
    test = pd.read_csv(CLEAN_DIR / "clean_test.csv")
    second_visit = train.iloc[[0]].copy()
    second_visit["MonthlyCharges"] = second_visit["MonthlyCharges"] + 1
    test = pd.concat([test, second_visit], ignore_index=True)
    train_path = tmp_path / "group_train.csv"
    test_path = tmp_path / "group_test.csv"
    output_path = tmp_path / "group_report.md"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    summary = run_leak_check(
        train_path=train_path,
        test_path=test_path,
        target_col=TARGET_COL,
        output_path=output_path,
        ignore_cols=[ID_COL],
    )

    assert summary["group_overlap_flagged"] == 1
    assert summary["group_column"] == ID_COL
    report = output_path.read_text(encoding="utf-8")
    assert "Shared entity groups" in report
    assert "Total unique test rows affected: **0**" not in report


def test_temporal_feature_leak_is_reported_end_to_end(tmp_path):
    output_path = tmp_path / "temporal_feature_report.md"
    summary = run_leak_check(
        train_path=CLEAN_DIR / ".." / "leaky" / "leaky_temporal_feature_train.csv",
        test_path=CLEAN_DIR / ".." / "leaky" / "leaky_temporal_feature_test.csv",
        target_col=TARGET_COL,
        output_path=output_path,
        ignore_cols=[ID_COL],
        label_date_col="label_date",
        feature_date_cols=["last_contact_date", "last_bill_date"],
    )

    assert summary["feature_timestamp_flagged"] > 0
    assert summary["split_chronology_flagged"] == 0
    assert "Temporal Leakage" in output_path.read_text(encoding="utf-8")


def test_temporal_split_leak_is_reported_end_to_end(tmp_path):
    output_path = tmp_path / "temporal_split_report.md"
    summary = run_leak_check(
        train_path=CLEAN_DIR / ".." / "leaky" / "leaky_temporal_split_train.csv",
        test_path=CLEAN_DIR / ".." / "leaky" / "leaky_temporal_split_test.csv",
        target_col=TARGET_COL,
        output_path=output_path,
        ignore_cols=[ID_COL],
        label_date_col="label_date",
        feature_date_cols=["last_contact_date", "last_bill_date"],
    )

    assert summary["feature_timestamp_flagged"] == 0
    assert summary["split_chronology_flagged"] > 0


def test_static_code_leakage_is_reported_end_to_end(tmp_path):
    code_path = tmp_path / "model.py"
    code_path.write_text("scaler.fit(X)\n", encoding="utf-8")
    output_path = tmp_path / "code_report.md"

    summary = run_leak_check(
        train_path=CLEAN_DIR / "clean_train.csv",
        test_path=CLEAN_DIR / "clean_test.csv",
        target_col=TARGET_COL,
        output_path=output_path,
        ignore_cols=[ID_COL],
        code_path=code_path,
    )

    assert summary["static_code_leakage_flagged"] == 1
    assert "Static Code Leakage" in output_path.read_text(encoding="utf-8")


def test_schema_mismatch_has_clear_error(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"Churn": [0, 1], "feature": [1, 2]}).to_csv(train_path, index=False)
    pd.DataFrame({"Churn": [0, 1], "other": [1, 2]}).to_csv(test_path, index=False)

    with pytest.raises(InputDataError, match="schemas do not match"):
        run_leak_check(train_path, test_path, target_col="Churn")


def test_json_output_contains_summary_and_results(tmp_path):
    output_path = tmp_path / "report.json"
    summary = run_leak_check(
        CLEAN_DIR / "clean_train.csv",
        CLEAN_DIR / "clean_test.csv",
        target_col=TARGET_COL,
        output_path=output_path,
        ignore_cols=[ID_COL],
        profile="overlap",
        output_format="json",
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["summary"]["report_path"] == str(output_path)
    assert report["metadata"]["profile"] == "overlap"
    assert report["results"]["exact"]["check"] == "exact_overlap"
    assert summary["target_predictiveness_flagged"] == 0


def test_target_profile_skips_overlap_checks(tmp_path):
    summary = run_leak_check(
        CLEAN_DIR / "clean_train.csv",
        CLEAN_DIR / "clean_test.csv",
        target_col=TARGET_COL,
        output_path=tmp_path / "target.md",
        profile="target",
    )

    assert summary["exact_overlap_flagged"] == 0
    assert summary["near_overlap_flagged"] == 0


def test_invalid_config_has_clear_error(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"unknown_option": true}', encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "leak-check",
            str(CLEAN_DIR / "clean_train.csv"),
            str(CLEAN_DIR / "clean_test.csv"),
            "--config",
            str(config_path),
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2


def test_input_row_limit_has_clear_error(tmp_path):
    with pytest.raises(InputDataError, match="exceeding the configured limit"):
        run_leak_check(
            CLEAN_DIR / "clean_train.csv",
            CLEAN_DIR / "clean_test.csv",
            target_col=TARGET_COL,
            max_input_rows=1,
        )


def test_report_file_is_written_and_nonempty(tmp_path):
    output_path = tmp_path / "report.md"
    run_leak_check(
        train_path=CLEAN_DIR / "clean_train.csv",
        test_path=CLEAN_DIR / "clean_test.csv",
        target_col=TARGET_COL,
        output_path=str(output_path),
        ignore_cols=[ID_COL],
    )
    content = Path(output_path).read_text(encoding="utf-8")
    assert "# Data Leakage Report" in content
    assert len(content) > 100


def test_missing_input_file_has_clear_error(tmp_path):
    with pytest.raises(InputDataError, match="Training file not found at:"):
        run_leak_check(
            train_path=tmp_path / "missing.csv",
            test_path=CLEAN_DIR / "clean_test.csv",
            target_col=TARGET_COL,
        )


def test_empty_input_file_has_clear_error(tmp_path):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")

    with pytest.raises(InputDataError, match="Training data is empty"):
        run_leak_check(
            train_path=empty_path,
            test_path=CLEAN_DIR / "clean_test.csv",
            target_col=TARGET_COL,
        )


def test_missing_target_column_has_clear_error(tmp_path):
    invalid_path = tmp_path / "missing_target.csv"
    pd.DataFrame({"feature": [1, 2, 3]}).to_csv(invalid_path, index=False)

    with pytest.raises(InputDataError, match="Target column 'Churn' not found in training data"):
        run_leak_check(
            train_path=invalid_path,
            test_path=CLEAN_DIR / "clean_test.csv",
            target_col=TARGET_COL,
        )


def test_cli_prints_clear_input_error_and_exits(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "leak-check",
            str(tmp_path / "missing.csv"),
            str(CLEAN_DIR / "clean_test.csv"),
            "--target",
            TARGET_COL,
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.err == f"Training file not found at: {tmp_path / 'missing.csv'}\n"
    assert captured.out == ""