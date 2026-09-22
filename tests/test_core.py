"""
test_core.py

End-to-end test: runs the whole pipeline through run_leak_check() and
checks the summary dict, exactly like a real user of the tool would.
"""

from pathlib import Path

import pandas as pd
import pytest

from leak_detector.core import InputDataError, run_leak_check
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
    with pytest.raises(InputDataError, match="Training CSV file not found"):
        run_leak_check(
            train_path=tmp_path / "missing.csv",
            test_path=CLEAN_DIR / "clean_test.csv",
            target_col=TARGET_COL,
        )


def test_empty_input_file_has_clear_error(tmp_path):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")

    with pytest.raises(InputDataError, match="Training CSV file is empty"):
        run_leak_check(
            train_path=empty_path,
            test_path=CLEAN_DIR / "clean_test.csv",
            target_col=TARGET_COL,
        )


def test_missing_target_column_has_clear_error(tmp_path):
    invalid_path = tmp_path / "missing_target.csv"
    pd.DataFrame({"feature": [1, 2, 3]}).to_csv(invalid_path, index=False)

    with pytest.raises(InputDataError, match="Target column 'Churn'"):
        run_leak_check(
            train_path=invalid_path,
            test_path=CLEAN_DIR / "clean_test.csv",
            target_col=TARGET_COL,
        )