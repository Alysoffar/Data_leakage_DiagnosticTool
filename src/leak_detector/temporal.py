"""Checks for feature timestamps and chronological train/test splits."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
	from .types import CheckResult


def _parse_date_column(df: pd.DataFrame, column: str) -> pd.Series:
	if column not in df.columns:
		raise ValueError(f"Date column '{column}' not found in DataFrame.")

	parsed = pd.to_datetime(df[column], errors="coerce")
	invalid = df[column].notna() & parsed.isna()
	if invalid.any():
		raise ValueError(f"Date column '{column}' contains invalid date values.")
	return parsed


def check_feature_timestamp_order(
	df: pd.DataFrame,
	label_date_col: str,
	feature_date_cols: Iterable[str],
	strict: bool = True,
) -> "CheckResult":
	"""Flag feature observations recorded after their row's label date."""
	label_dates = _parse_date_column(df, label_date_col)
	feature_date_cols = list(feature_date_cols)
	details = []

	for column in feature_date_cols:
		feature_dates = _parse_date_column(df, column)
		comparable = label_dates.notna() & feature_dates.notna()
		violations = feature_dates >= label_dates if strict else feature_dates > label_dates
		for row_idx in df.index[comparable & violations]:
			details.append(
				{
					"row_idx": row_idx,
					"column": column,
					"label_date": label_dates.loc[row_idx],
					"feature_date": feature_dates.loc[row_idx],
				}
			)

	return {
		"check": "feature_timestamp_order",
		"n_flagged": len(details),
		"detail": details,
	}


def check_split_chronology(train_df: pd.DataFrame, test_df: pd.DataFrame, date_col: str) -> "CheckResult":
	"""Flag training rows dated after the earliest test row."""
	train_dates = _parse_date_column(train_df, date_col).dropna()
	test_dates = _parse_date_column(test_df, date_col).dropna()

	if train_dates.empty or test_dates.empty:
		return {
			"check": "split_chronology",
			"n_flagged": 0,
			"detail": {
				"train_max_date": None,
				"test_min_date": None,
				"is_chronological": True,
			},
		}

	train_max = train_dates.max()
	test_min = test_dates.min()
	flagged = int((train_dates > test_min).sum())
	return {
		"check": "split_chronology",
		"n_flagged": flagged,
		"detail": {
			"train_max_date": train_max,
			"test_min_date": test_min,
			"is_chronological": bool(train_max <= test_min),
		},
	}
