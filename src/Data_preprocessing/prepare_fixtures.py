"""Build clean and intentionally leaky train/test datasets from one CSV.

Produces 14 files in ``Data/processed/<dataset>/``:
  clean_train.csv / clean_test.csv
      -> a normal, leak-free split. Your tool should find NOTHING here.

  leaky_overlap_train.csv / leaky_overlap_test.csv
      -> same split, but 5% of the test rows have been copied into
         training. Your overlap checker (Step 3) should catch this.

  leaky_target_train.csv / leaky_target_test.csv
      -> same split, but with an extra column "cheat_score" that's
         just the target plus a little noise. Your target-leakage
         checker (Step 4) should catch this.

    leaky_group_overlap_train.csv / leaky_group_overlap_test.csv
            -> a small set of training rows has a synthetic second visit in
                 test, retaining the same ID while changing other values.
                 A group-overlap checker should catch the shared entities.

      clean_temporal_train.csv / clean_temporal_test.csv
          -> date-aware chronological split with valid feature timestamps.

      leaky_temporal_feature_train.csv / leaky_temporal_feature_test.csv
          -> chronological split with feature timestamps after the label date.

      leaky_temporal_split_train.csv / leaky_temporal_split_test.csv
          -> valid feature timestamps but train/test rows out of chronological order.

Usage:
    python prepare_fixtures.py path/to/dataset.csv --target-column income --dataset-name adult
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leak_detector.paths import ProjectPaths


class FixtureBuilder:
    """Create clean and intentionally leaky datasets in one data directory."""

    def __init__(self, paths: ProjectPaths, dataset_name: str | None = None):
        self.paths = paths
        self.dataset_name = dataset_name

    def build(self, raw_path: str | Path, target_column: str) -> None:
        self.paths.ensure_output_directories()
        df = pd.read_csv(raw_path)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in {raw_path}.")

        if df[target_column].nunique(dropna=True) != 2:
            raise ValueError(f"Target column '{target_column}' must contain exactly two classes.")

        # Encode any binary target so the synthetic target-leakage feature is numeric.
        df[target_column] = pd.factorize(df[target_column])[0]
        dataset_root = self.paths.dataset_data(self.dataset_name)
        clean_dir = dataset_root / "clean"
        leaky_dir = dataset_root / "leaky"
        clean_dir.mkdir(parents=True, exist_ok=True)
        leaky_dir.mkdir(parents=True, exist_ok=True)

        train, test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df[target_column]
        )
        train.to_csv(clean_dir / "clean_train.csv", index=False)
        test.to_csv(clean_dir / "clean_test.csv", index=False)

        contaminated = test.sample(frac=0.05, random_state=1)
        leaky_overlap_train = pd.concat([train, contaminated], ignore_index=True)
        leaky_overlap_train.to_csv(leaky_dir / "leaky_overlap_train.csv", index=False)
        test.to_csv(leaky_dir / "leaky_overlap_test.csv", index=False)

        rng = np.random.default_rng(42)
        leaky_target_train = train.copy()
        leaky_target_test = test.copy()
        for split in (leaky_target_train, leaky_target_test):
            split["cheat_score"] = split[target_column] + rng.normal(0, 0.05, size=len(split))
        leaky_target_train.to_csv(leaky_dir / "leaky_target_train.csv", index=False)
        leaky_target_test.to_csv(leaky_dir / "leaky_target_test.csv", index=False)

        group_column = self._find_group_column(df) or "fixture_group_id"
        self.build_group_overlap_fixture(
            train,
            test,
            group_column=group_column,
            output_dir=leaky_dir,
            target_column=target_column,
        )
        self.build_temporal_fixtures(df, output_root=dataset_root)

    @staticmethod
    def _find_group_column(df: pd.DataFrame) -> str | None:
        """Return a likely entity identifier for the group-overlap fixture."""
        candidates = ("customerID", "customer_id", "account_id", "id")
        for column in candidates:
            if column in df.columns and df[column].nunique(dropna=True) > 1:
                return column
        return None

    @staticmethod
    def build_group_overlap_fixture(
        train: pd.DataFrame,
        test: pd.DataFrame,
        group_column: str,
        output_dir: str | Path,
        sample_size: int = 5,
        target_column: str | None = None,
    ) -> None:
        """Write a fixture with related visits for entities in both splits.

        The generated test rows retain the selected entities' IDs but have
        deterministic changes to their non-ID values, so this is group
        leakage rather than exact row duplication.
        """
        if sample_size < 1:
            raise ValueError("sample_size must be at least 1.")

        train = train.copy()
        test = test.copy()
        if group_column not in train.columns and group_column not in test.columns:
            train[group_column] = [f"train_{index}" for index in train.index]
            test[group_column] = [f"test_{index}" for index in test.index]
        elif group_column not in train.columns or group_column not in test.columns:
            raise ValueError(f"Group column '{group_column}' must exist in both splits.")

        source_rows = train.drop_duplicates(subset=[group_column]).head(sample_size).copy()
        if source_rows.empty:
            raise ValueError("Cannot build a group-overlap fixture from an empty training split.")

        visit_rows = source_rows.copy()
        for column in visit_rows.columns:
            if column in {group_column, target_column}:
                continue
            if pd.api.types.is_numeric_dtype(visit_rows[column]):
                visit_rows[column] = visit_rows[column] + 1
            else:
                visit_rows[column] = visit_rows[column].astype(str) + "_visit2"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(output_dir / "leaky_group_overlap_train.csv", index=False)
        pd.concat([test, visit_rows], ignore_index=True).to_csv(
            output_dir / "leaky_group_overlap_test.csv", index=False
        )

    @staticmethod
    def build_temporal_fixtures(df: pd.DataFrame, output_root: str | Path) -> None:
        """Write clean, feature-leaky, and split-leaky temporal fixtures."""
        if len(df) < 10:
            raise ValueError("At least 10 rows are required for temporal fixtures.")

        dated = df.reset_index(drop=True).copy()
        label_dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(dated.index, unit="D")
        dated["label_date"] = label_dates.strftime("%Y-%m-%d")
        dated["last_contact_date"] = (label_dates - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        dated["last_bill_date"] = (label_dates - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        split_at = int(len(dated) * 0.8)
        clean_train = dated.iloc[:split_at].copy()
        clean_test = dated.iloc[split_at:].copy()
        clean_dir = Path(output_root) / "clean"
        leaky_dir = Path(output_root) / "leaky"
        clean_dir.mkdir(parents=True, exist_ok=True)
        leaky_dir.mkdir(parents=True, exist_ok=True)
        clean_train.to_csv(clean_dir / "clean_temporal_train.csv", index=False)
        clean_test.to_csv(clean_dir / "clean_temporal_test.csv", index=False)

        leaky_feature_train = clean_train.copy()
        feature_rows = leaky_feature_train.index[: min(5, len(leaky_feature_train))]
        feature_dates = pd.to_datetime(leaky_feature_train.loc[feature_rows, "label_date"])
        leaky_feature_train.loc[feature_rows, "last_contact_date"] = (
            feature_dates + pd.Timedelta(days=2)
        ).dt.strftime("%Y-%m-%d")
        leaky_feature_train.to_csv(leaky_dir / "leaky_temporal_feature_train.csv", index=False)
        clean_test.to_csv(leaky_dir / "leaky_temporal_feature_test.csv", index=False)

        leaky_split_train = clean_train.copy()
        leaky_split_test = clean_test.copy()
        swap_count = min(5, len(leaky_split_train), len(leaky_split_test))
        train_tail = leaky_split_train.tail(swap_count).copy()
        test_head = leaky_split_test.head(swap_count).copy()
        leaky_split_train.iloc[-swap_count:] = test_head.to_numpy()
        leaky_split_test.iloc[:swap_count] = train_tail.to_numpy()
        leaky_split_train.to_csv(leaky_dir / "leaky_temporal_split_train.csv", index=False)
        leaky_split_test.to_csv(leaky_dir / "leaky_temporal_split_test.csv", index=False)


def main(raw_path: str, target_column: str = "Churn", dataset_name: str | None = None) -> None:
    paths = ProjectPaths.from_file(__file__)
    FixtureBuilder(paths, dataset_name).build(raw_path, target_column)
    print(f"Done. Wrote generated datasets to {paths.dataset_data(dataset_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare clean and leaky dataset variants.")
    parser.add_argument("raw_path", help="Path to the source CSV")
    parser.add_argument("--target-column", default="Churn", help="Binary target column")
    parser.add_argument("--dataset-name", help="Subdirectory name under Data/processed")
    args = parser.parse_args()
    main(args.raw_path, args.target_column, args.dataset_name)