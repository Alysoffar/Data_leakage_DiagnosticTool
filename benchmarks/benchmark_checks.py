"""Small repeatable benchmark for the most expensive dataset checks."""

from pathlib import Path
from time import perf_counter

import pandas as pd
from leak_detector.overlap import check_near_duplicates
from leak_detector.target_leakage import check_single_feature_predictiveness

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "Data" / "processed" / "clean" / "clean_train.csv"
TEST = ROOT / "Data" / "processed" / "clean" / "clean_test.csv"


def main() -> None:
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    for name, callback in (
        (
            "near_duplicates",
            lambda: check_near_duplicates(
                train, test, ignore_cols=["customerID"], target_col="Churn", max_rows=1000
            ),
        ),
        (
            "target_predictiveness",
            lambda: check_single_feature_predictiveness(train, "Churn"),
        ),
    ):
        started = perf_counter()
        result = callback()
        elapsed = perf_counter() - started
        print(f"{name}: {elapsed:.3f}s, flagged={result['n_flagged']}")


if __name__ == "__main__":
    main()
