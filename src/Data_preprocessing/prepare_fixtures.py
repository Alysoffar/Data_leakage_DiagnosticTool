"""Build clean and intentionally leaky train/test datasets from one CSV.

Produces 6 files in ``Data/processed/<dataset>/``:
  clean_train.csv / clean_test.csv
      -> a normal, leak-free split. Your tool should find NOTHING here.

  leaky_overlap_train.csv / leaky_overlap_test.csv
      -> same split, but 5% of the test rows have been copied into
         training. Your overlap checker (Step 3) should catch this.

  leaky_target_train.csv / leaky_target_test.csv
      -> same split, but with an extra column "cheat_score" that's
         just the target plus a little noise. Your target-leakage
         checker (Step 4) should catch this.

Usage:
    python prepare_fixtures.py path/to/dataset.csv --target-column income --dataset-name adult
"""

from pathlib import Path
import sys
import argparse

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