"""
core.py

The single entry point for the leak-detector tool. Ties together the
overlap checks (overlap.py), the target-leakage checks (target_leakage.py),
and the report builder (report.py) into one function -- and one CLI
command.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from .overlap import check_exact_duplicates, check_near_duplicates
    from .paths import ProjectPaths
    from .report import generate_full_report
    from .target_leakage import check_correlation, check_single_feature_predictiveness
except ImportError:  # Supports running this file directly from src/leak_detector.
    from overlap import check_exact_duplicates, check_near_duplicates
    from paths import ProjectPaths
    from report import generate_full_report
    from target_leakage import check_correlation, check_single_feature_predictiveness


@dataclass(frozen=True)
class LeakageDiagnostic:
    """Configured leakage checks that can be reused across dataset pairs."""

    target_col: str
    ignore_cols: list[str] | None = None
    overlap_threshold: float = 0.95
    target_threshold: float = 0.9

    def run(
        self,
        train_path: str | Path,
        test_path: str | Path,
        output_path: str | Path | None = None,
    ) -> dict:
        """Run all checks and write one combined Markdown report."""
        paths = ProjectPaths.from_file(__file__)
        if output_path is None:
            output_path = paths.report_path()
        elif not Path(output_path).is_absolute():
            output_path = paths.report_path(Path(output_path))

        train_df = _read_input_csv(train_path, "Training")
        test_df = _read_input_csv(test_path, "Test")

        if self.target_col not in train_df.columns:
            raise InputDataError(
                f"Target column '{self.target_col}' was not found in the training CSV."
            )
        if self.target_col not in test_df.columns:
            raise InputDataError(
                f"Target column '{self.target_col}' was not found in the test CSV."
            )

        exact_result = check_exact_duplicates(train_df, test_df, ignore_cols=self.ignore_cols)
        near_result = check_near_duplicates(train_df, test_df, threshold=self.overlap_threshold)
        predictiveness_result = check_single_feature_predictiveness(
            train_df, target_col=self.target_col, threshold=self.target_threshold
        )
        correlation_result = check_correlation(
            train_df, target_col=self.target_col, threshold=self.target_threshold
        )

        generate_full_report(
            train_df, test_df, self.target_col,
            exact_result, near_result,
            predictiveness_result, correlation_result,
            output_path,
        )

        return {
            "exact_overlap_flagged": exact_result["n_flagged"],
            "near_overlap_flagged": near_result["n_flagged"],
            "target_predictiveness_flagged": predictiveness_result["n_flagged"],
            "target_correlation_flagged": correlation_result["n_flagged"],
            "report_path": output_path,
        }


def run_leak_check(
    train_path: str,
    test_path: str,
    target_col: str,
    output_path: str | Path | None = None,
    ignore_cols: list[str] | None = None,
    overlap_threshold: float = 0.95,
    target_threshold: float = 0.9,
) -> dict:
    """Backward-compatible function wrapper around :class:`LeakageDiagnostic`."""
    return LeakageDiagnostic(
        target_col=target_col,
        ignore_cols=ignore_cols,
        overlap_threshold=overlap_threshold,
        target_threshold=target_threshold,
    ).run(train_path, test_path, output_path)


class InputDataError(ValueError):
    """Raised when an input dataset cannot be analyzed."""


def _read_input_csv(path: str | Path, label: str) -> pd.DataFrame:
    """Read one non-empty input CSV and convert I/O failures to clear messages."""
    input_path = Path(path)
    try:
        dataframe = pd.read_csv(input_path)
    except FileNotFoundError as exc:
        raise InputDataError(f"{label} CSV file not found: {input_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise InputDataError(f"{label} CSV file is empty: {input_path}") from exc
    except pd.errors.ParserError as exc:
        raise InputDataError(f"Could not parse {label.lower()} CSV: {input_path}") from exc

    if dataframe.empty:
        raise InputDataError(f"{label} CSV contains no data rows: {input_path}")
    return dataframe


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leak-check",
        description="Check a train/test dataset pair for data leakage.",
    )
    parser.add_argument("train_path", help="Path to the training CSV")
    parser.add_argument("test_path", help="Path to the test CSV")
    parser.add_argument("--target", required=True, help="Name of the target column")
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
        "--overlap-threshold", type=float, default=0.95, help="Near-duplicate similarity threshold (0-1)"
    )
    parser.add_argument(
        "--target-threshold", type=float, default=0.9, help="Single-feature AUC/correlation threshold (0-1)"
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        summary = run_leak_check(
            train_path=args.train_path,
            test_path=args.test_path,
            target_col=args.target,
            output_path=args.output,
            ignore_cols=args.ignore_cols,
            overlap_threshold=args.overlap_threshold,
            target_threshold=args.target_threshold,
        )
    except (InputDataError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Report written to {summary['report_path']}")
    print(
        f"exact_overlap={summary['exact_overlap_flagged']} "
        f"near_overlap={summary['near_overlap_flagged']} "
        f"target_predictiveness={summary['target_predictiveness_flagged']} "
        f"target_correlation={summary['target_correlation_flagged']}"
    )


if __name__ == "__main__":
    main()