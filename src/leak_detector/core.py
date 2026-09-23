"""
core.py

The single entry point for the leak-detector tool. Ties together the
overlap checks (overlap.py), the target-leakage checks (target_leakage.py),
and the report builder (report.py) into one function -- and one CLI
command.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .types import CheckResult

try:
    from .overlap import check_exact_duplicates, check_group_overlap, check_near_duplicates
    from .paths import ProjectPaths
    from .pipeline_checker import check_code_leakage_extended
    from .report import generate_full_report
    from .target_leakage import check_correlation, check_single_feature_predictiveness
    from .temporal import check_feature_timestamp_order, check_split_chronology
    from .types import REPORT_SCHEMA_VERSION
except ImportError:  # Supports running this file directly from src/leak_detector.
    from overlap import check_exact_duplicates, check_group_overlap, check_near_duplicates
    from paths import ProjectPaths
    from pipeline_checker import check_code_leakage_extended
    from report import generate_full_report
    from target_leakage import check_correlation, check_single_feature_predictiveness
    from temporal import check_feature_timestamp_order, check_split_chronology
    REPORT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class LeakageDiagnostic:
    """Configured leakage checks that can be reused across dataset pairs."""

    target_col: str
    ignore_cols: list[str] | None = None
    group_col: str | None = None
    label_date_col: str | None = None
    feature_date_cols: list[str] | None = None
    temporal_strict: bool = True
    code_path: str | Path | None = None
    overlap_threshold: float = 0.95
    target_threshold: float = 0.9
    profile: str = "full"
    output_format: str = "markdown"
    near_max_rows: int | None = None
    max_input_rows: int | None = None

    def run(
        self,
        train_path: str | Path,
        test_path: str | Path,
        output_path: str | Path | None = None,
    ) -> dict:
        """Run all checks and write one combined Markdown report."""
        if self.profile not in {"full", "overlap", "target", "temporal", "code"}:
            raise InputDataError(f"Unknown check profile '{self.profile}'.")
        if self.output_format not in {"markdown", "json"}:
            raise InputDataError(f"Unknown output format '{self.output_format}'.")
        paths = ProjectPaths.from_file(__file__)
        if output_path is None:
            output_path = paths.report_path()
        elif not Path(output_path).is_absolute():
            output_path = paths.report_path(Path(output_path))

        train_df = _read_input_csv(train_path, "Training", self.max_input_rows)
        test_df = _read_input_csv(test_path, "Test", self.max_input_rows)

        if self.target_col not in train_df.columns:
            raise InputDataError(
                f"Target column '{self.target_col}' not found in training data."
            )
        if self.target_col not in test_df.columns:
            raise InputDataError(
                f"Target column '{self.target_col}' not found in test data."
            )
        _validate_schema(train_df, test_df)

        exact_result: CheckResult = {"check": "exact_overlap", "n_flagged": 0, "detail": []}
        near_result: CheckResult = {"check": "near_duplicate", "n_flagged": 0, "detail": []}
        group_result: CheckResult = {"check": "group_overlap", "group_col": None, "n_flagged": 0, "detail": []}
        if self.profile in {"full", "overlap"}:
            exact_result = check_exact_duplicates(train_df, test_df, ignore_cols=self.ignore_cols)
            near_result = check_near_duplicates(
                train_df,
                test_df,
                threshold=self.overlap_threshold,
                ignore_cols=self.ignore_cols,
                target_col=self.target_col,
                max_rows=self.near_max_rows,
            )
            resolved_group_col = _resolve_group_column(train_df, test_df, self.group_col)
            if resolved_group_col:
                group_result = check_group_overlap(train_df, test_df, resolved_group_col)

        predictiveness_result: CheckResult = {"check": "single_feature_predictiveness", "n_flagged": 0, "detail": []}
        correlation_result: CheckResult = {"check": "correlation", "n_flagged": 0, "detail": []}
        if self.profile in {"full", "target"}:
            predictiveness_result = check_single_feature_predictiveness(
                train_df, target_col=self.target_col, threshold=self.target_threshold
            )
            correlation_result = check_correlation(
                train_df, target_col=self.target_col, threshold=self.target_threshold
            )

        if self.profile in {"full", "temporal"}:
            feature_timestamp_result, split_chronology_result = _run_temporal_checks(
                train_df, test_df, self.label_date_col, self.feature_date_cols, self.temporal_strict
            )
        else:
            feature_timestamp_result, split_chronology_result = _empty_temporal_results()
        code_result: CheckResult = _run_code_check(self.code_path) if self.profile in {"full", "code"} else _skipped_code_result()

        results = {
            "exact": exact_result,
            "near": near_result,
            "group": group_result,
            "predictiveness": predictiveness_result,
            "correlation": correlation_result,
            "feature_timestamp": feature_timestamp_result,
            "split_chronology": split_chronology_result,
            "static_code": code_result,
        }
        metadata = {
            "profile": self.profile,
            "schema_version": REPORT_SCHEMA_VERSION,
            "output_format": self.output_format,
            "train_path": str(train_path),
            "test_path": str(test_path),
            "target_column": self.target_col,
            "ignore_columns": self.ignore_cols or [],
            "group_column": group_result.get("group_col"),
            "overlap_threshold": self.overlap_threshold,
            "target_threshold": self.target_threshold,
            "max_input_rows": self.max_input_rows,
        }
        summary = {
            "exact_overlap_flagged": exact_result["n_flagged"],
            "near_overlap_flagged": near_result["n_flagged"],
            "group_overlap_flagged": group_result["n_flagged"],
            "group_column": group_result["group_col"],
            "target_predictiveness_flagged": predictiveness_result["n_flagged"],
            "target_correlation_flagged": correlation_result["n_flagged"],
            "feature_timestamp_flagged": feature_timestamp_result["n_flagged"],
            "split_chronology_flagged": split_chronology_result["n_flagged"],
            "static_code_leakage_flagged": code_result["n_flagged"],
            "report_path": output_path,
            "metadata": metadata,
        }
        if self.output_format == "json":
            _write_json_report(output_path, summary, results, metadata)
        else:
            generate_full_report(
                train_df, test_df, self.target_col,
                exact_result, near_result, predictiveness_result, correlation_result,
                output_path,
                group_result=group_result,
                feature_timestamp_result=feature_timestamp_result,
                split_chronology_result=split_chronology_result,
                code_result=code_result,
                metadata=metadata,
            )
        return summary


def run_leak_check(
    train_path: str,
    test_path: str,
    target_col: str,
    output_path: str | Path | None = None,
    ignore_cols: list[str] | None = None,
    group_col: str | None = None,
    label_date_col: str | None = None,
    feature_date_cols: list[str] | None = None,
    temporal_strict: bool = True,
    code_path: str | Path | None = None,
    overlap_threshold: float = 0.95,
    target_threshold: float = 0.9,
    profile: str = "full",
    output_format: str = "markdown",
    near_max_rows: int | None = None,
    max_input_rows: int | None = None,
) -> dict:
    """Backward-compatible function wrapper around :class:`LeakageDiagnostic`."""
    return LeakageDiagnostic(
        target_col=target_col,
        ignore_cols=ignore_cols,
        group_col=group_col,
        label_date_col=label_date_col,
        feature_date_cols=feature_date_cols,
        temporal_strict=temporal_strict,
        code_path=code_path,
        overlap_threshold=overlap_threshold,
        target_threshold=target_threshold,
        profile=profile,
        output_format=output_format,
        near_max_rows=near_max_rows,
        max_input_rows=max_input_rows,
    ).run(train_path, test_path, output_path)


class InputDataError(ValueError):
    """Raised when an input dataset cannot be analyzed."""


def _read_input_csv(path: str | Path, label: str, max_rows: int | None = None) -> pd.DataFrame:
    """Read one non-empty input CSV and convert I/O failures to clear messages."""
    input_path = Path(path)
    try:
        dataframe = pd.read_csv(input_path)
    except FileNotFoundError as exc:
        raise InputDataError(f"{label} file not found at: {input_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise InputDataError(f"{label} data is empty") from exc
    except pd.errors.ParserError as exc:
        raise InputDataError(f"Could not parse {label.lower()} CSV: {input_path}") from exc

    if dataframe.empty:
        raise InputDataError(f"{label} data is empty")
    if max_rows is not None:
        if max_rows < 1:
            raise InputDataError("max_input_rows must be at least 1 when provided.")
        if len(dataframe) > max_rows:
            raise InputDataError(
                f"{label} data has {len(dataframe)} rows, exceeding the configured limit of {max_rows}."
            )
    return dataframe


def _validate_schema(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Reject train/test pairs whose columns cannot be compared safely."""
    if train_df.columns.duplicated().any() or test_df.columns.duplicated().any():
        raise InputDataError("Training and test schemas cannot contain duplicate column names.")
    train_columns = list(train_df.columns)
    test_columns = list(test_df.columns)
    missing_from_test = [column for column in train_columns if column not in test_df.columns]
    missing_from_train = [column for column in test_columns if column not in train_df.columns]
    if missing_from_test or missing_from_train:
        parts = []
        if missing_from_test:
            parts.append(f"missing from test: {missing_from_test}")
        if missing_from_train:
            parts.append(f"missing from training: {missing_from_train}")
        raise InputDataError("Training and test schemas do not match (" + "; ".join(parts) + ").")


def _empty_temporal_results() -> tuple["CheckResult", "CheckResult"]:
    feature_result: CheckResult = {"check": "feature_timestamp_order", "n_flagged": 0, "detail": []}
    split_result: CheckResult = {
        "check": "split_chronology",
        "n_flagged": 0,
        "detail": {"train_max_date": None, "test_min_date": None, "is_chronological": True},
    }
    return feature_result, split_result


def _skipped_code_result() -> "CheckResult":
    return {"check": "static_code_leakage", "n_flagged": 0, "detail": [], "skipped": True}


def _write_json_report(output_path: str | Path, summary: dict, results: dict, metadata: dict) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(
            {"schema_version": REPORT_SCHEMA_VERSION, "metadata": metadata, "summary": summary, "results": results},
            default=str,
            indent=2,
        ),
        encoding="utf-8",
    )


def _resolve_group_column(
    train_df: pd.DataFrame, test_df: pd.DataFrame, requested: str | None
) -> str | None:
    """Choose an entity column for group-overlap checks, if one is available."""
    if requested:
        if requested not in train_df.columns or requested not in test_df.columns:
            raise InputDataError(f"Group column '{requested}' was not found in both CSV files.")
        return requested

    candidates = ("fixture_group_id", "customerID", "customer_id", "account_id", "patient_id", "user_id")
    return next(
        (column for column in candidates if column in train_df.columns and column in test_df.columns),
        None,
    )


def _run_temporal_checks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_date_col: str | None,
    feature_date_cols: list[str] | None,
    strict: bool,
) -> tuple["CheckResult", "CheckResult"]:
    """Run temporal checks only when the caller supplies temporal columns."""
    empty_feature_result: CheckResult = {
        "check": "feature_timestamp_order",
        "n_flagged": 0,
        "detail": [],
    }
    empty_split_result: CheckResult = {
        "check": "split_chronology",
        "n_flagged": 0,
        "detail": {
            "train_max_date": None,
            "test_min_date": None,
            "is_chronological": True,
        },
    }
    if label_date_col is None and not feature_date_cols:
        return empty_feature_result, empty_split_result
    if not label_date_col or not feature_date_cols:
        raise InputDataError(
            "Temporal checks require both --label-date-col and --feature-date-cols."
        )

    required_columns = [label_date_col, *feature_date_cols]
    for column in required_columns:
        if column not in train_df.columns:
            raise InputDataError(f"Temporal column '{column}' not found in training data.")
        if column not in test_df.columns:
            raise InputDataError(f"Temporal column '{column}' not found in test data.")

    feature_result = check_feature_timestamp_order(
        train_df, label_date_col, feature_date_cols, strict=strict
    )
    feature_test_result = check_feature_timestamp_order(
        test_df, label_date_col, feature_date_cols, strict=strict
    )
    feature_result["detail"].extend(feature_test_result["detail"])
    feature_result["n_flagged"] += feature_test_result["n_flagged"]
    split_result = check_split_chronology(train_df, test_df, label_date_col)
    feature_result["label_date_col"] = label_date_col
    split_result["detail"]["date_col"] = label_date_col
    return feature_result, split_result


def _run_code_check(code_path: str | Path | None) -> "CheckResult":
    """Run the optional static checker against a user-provided Python file."""
    if code_path is None:
        return {"check": "static_code_leakage", "n_flagged": 0, "detail": [], "skipped": True}

    source_path = Path(code_path)
    try:
        source = source_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise InputDataError(f"Code file not found at: {source_path}") from exc
    except OSError as exc:
        raise InputDataError(f"Could not read code file at: {source_path}") from exc
    return check_code_leakage_extended(source)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leak-check",
        description="Check a train/test dataset pair for data leakage.",
    )
    parser.add_argument("train_path", help="Path to the training CSV")
    parser.add_argument("test_path", help="Path to the test CSV")
    parser.add_argument("--target", default=None, help="Name of the target column")
    parser.add_argument("--config", default=None, help="Optional JSON configuration file")
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the Markdown report (default: reports/leak_report.md)",
    )
    parser.add_argument(
        "--ignore-cols",
        nargs="*",
        default=None,
        help="Columns to exclude from duplicate detection (e.g. an ID column)",
    )
    parser.add_argument(
        "--group-col",
        default=None,
        help="Entity column used to detect groups appearing in both train and test",
    )
    parser.add_argument("--label-date-col", default=None, help="Label/outcome date column for temporal checks")
    parser.add_argument(
        "--feature-date-cols",
        nargs="*",
        default=None,
        help="Feature date columns checked against the label date",
    )
    parser.add_argument(
        "--temporal-nonstrict",
        action="store_false",
        dest="temporal_strict",
        help="Allow feature dates equal to the label date",
    )
    parser.add_argument(
        "--overlap-threshold", type=float, default=0.95, help="Near-duplicate similarity threshold (0-1)"
    )
    parser.add_argument(
        "--target-threshold", type=float, default=0.9, help="Single-feature AUC/correlation threshold (0-1)"
    )
    parser.add_argument(
        "--code-path",
        default=None,
        help="Optional Python source file to scan for preprocessing leakage",
    )
    parser.add_argument(
        "--profile",
        choices=("full", "overlap", "target", "temporal", "code"),
        default="full",
        help="Checks to run (default: full)",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("markdown", "json"),
        default="markdown",
        help="Report format (default: markdown)",
    )
    parser.add_argument(
        "--near-max-rows",
        type=int,
        default=None,
        help="Maximum rows sampled per split for near-duplicate checks",
    )
    parser.add_argument(
        "--max-input-rows",
        type=int,
        default=None,
        help="Reject input CSVs larger than this row count",
    )
    return parser


def _load_config(path: str | Path | None) -> dict:
    if path is None:
        return {}
    config_path = Path(path)
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputDataError(f"Config file not found at: {config_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise InputDataError(f"Could not read JSON config at: {config_path}") from exc


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        config = _load_config(args.config)
        for name, value in config.items():
            if not hasattr(args, name):
                raise InputDataError(f"Unknown config option '{name}'.")
            if getattr(args, name) == parser.get_default(name):
                setattr(args, name, value)
        if not args.target:
            raise InputDataError("Target column is required via --target or config.")
        summary = run_leak_check(
            train_path=args.train_path,
            test_path=args.test_path,
            target_col=args.target,
            output_path=args.output,
            ignore_cols=args.ignore_cols,
            group_col=args.group_col,
            label_date_col=args.label_date_col,
            feature_date_cols=args.feature_date_cols,
            temporal_strict=args.temporal_strict,
            code_path=args.code_path,
            profile=args.profile,
            output_format=args.output_format,
            near_max_rows=args.near_max_rows,
            max_input_rows=args.max_input_rows,
            overlap_threshold=args.overlap_threshold,
            target_threshold=args.target_threshold,
        )
    except InputDataError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

    print(f"Report written to {summary['report_path']}")
    print(
        f"exact_overlap={summary['exact_overlap_flagged']} "
        f"near_overlap={summary['near_overlap_flagged']} "
        f"group_overlap={summary['group_overlap_flagged']} "
        f"feature_timestamp={summary['feature_timestamp_flagged']} "
        f"split_chronology={summary['split_chronology_flagged']} "
        f"static_code_leakage={summary['static_code_leakage_flagged']} "
        f"target_predictiveness={summary['target_predictiveness_flagged']} "
        f"target_correlation={summary['target_correlation_flagged']}"
    )
    has_findings = any(
        summary[key] > 0
        for key in (
            "exact_overlap_flagged",
            "near_overlap_flagged",
            "group_overlap_flagged",
            "feature_timestamp_flagged",
            "split_chronology_flagged",
            "static_code_leakage_flagged",
            "target_predictiveness_flagged",
            "target_correlation_flagged",
        )
    )
    raise SystemExit(1 if has_findings else 0)


if __name__ == "__main__":
    main()