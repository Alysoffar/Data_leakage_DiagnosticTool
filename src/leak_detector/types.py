"""Shared result types for leakage checks."""

from typing import Any, Literal, TypedDict

CheckProfile = Literal["full", "overlap", "target", "temporal", "code"]
ReportFormat = Literal["markdown", "json"]
REPORT_SCHEMA_VERSION = "1.0"


class CheckError(TypedDict):
    """A recoverable issue encountered while evaluating one feature/check."""

    column: str
    error: str


class CodeViolation(TypedDict):
    """A static source location and explanation for a code finding."""

    line: int
    column: int
    method: str
    variable: str
    issue: str


class CheckResult(TypedDict, total=False):
    """Common shape returned by a detector check."""

    check: str
    n_flagged: int
    detail: Any
    errors: list[dict[str, str]]
    skipped: bool
    error: str
    scoring: str
    group_col: str | None
    test_indices: list[Any]
    label_date_col: str
    date_col: str


class ExtendedCheckResult(CheckResult):
    """Named alias for results that include optional metadata."""