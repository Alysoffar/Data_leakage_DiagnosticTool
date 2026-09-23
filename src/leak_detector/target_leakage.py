from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if TYPE_CHECKING:
    from .types import CheckResult


def is_id_like(feature_series: pd.Series, id_cardinality_threshold: float = 0.95) -> bool:
    """Return whether a feature has unusually high cardinality for an ID heuristic.

    This is intentionally a heuristic: high-cardinality numeric features can be
    valid measurements, so callers should treat skipped columns as candidates
    for review rather than proof that a column is an identifier.
    """
    total_obs = len(feature_series)
    if total_obs == 0:
        return False

    n_unique = feature_series.nunique(dropna=True)
    if n_unique / total_obs <= id_cardinality_threshold:
        return False

    if (
        pd.api.types.is_string_dtype(feature_series)
        or isinstance(feature_series.dtype, pd.CategoricalDtype)
    ):
        return True

    # Numeric measurements can naturally be high-cardinality. Restrict the
    # numeric branch to integer-like monotonic keys to avoid hiding leaked
    # continuous features such as a noisy target proxy.
    return bool(
        pd.api.types.is_integer_dtype(feature_series)
        and (feature_series.is_monotonic_increasing or feature_series.is_monotonic_decreasing)
    )


def check_single_feature_predictiveness(
    df: pd.DataFrame,
    target_col: str,
    threshold: float = 0.9,
    cv: int = 5,
    id_cardinality_threshold: float = 0.95,
) -> "CheckResult":
    """Evaluates each feature's standalone predictive power using CV AUC score

    to detect potential target leakage. Supports both binary targets (plain
    ROC AUC) and multi-class targets (one-vs-rest ROC AUC, averaged).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Drop rows where target is missing
    clean_df = df.dropna(subset=[target_col]).copy()
    y = clean_df[target_col]

    # Target must have at least 2 classes -- anything less can't be modeled.
    n_classes = y.nunique()
    if n_classes < 2:
        raise ValueError(f"Target '{target_col}' must have at least 2 classes.")

    # Multi-class ROC AUC needs enough examples per class to build `cv`
    # stratified folds. Fail loudly and specifically here, rather than
    # letting every single feature silently fail one by one below.
    class_counts = y.value_counts()
    if class_counts.min() < cv:
        raise ValueError(
            f"Target '{target_col}' has a class with only {class_counts.min()} "
            f"example(s), which is fewer than cv={cv} folds. Reduce cv or "
            "address the rare class before running this check."
        )

    # Binary keeps using plain "roc_auc". 3+ classes switches to one-vs-rest
    # AUC, which LogisticRegression(solver="liblinear") supports natively
    # since liblinear only ever does one-vs-rest for multi-class anyway.
    scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr"

    detail = []
    errors = []

    for col in clean_df.columns:
        if col == target_col:
            continue

        feature_series = clean_df[col]
        n_unique = feature_series.nunique(dropna=True)

        # 1. Skip constant or all-null columns
        if n_unique <= 1:
            continue

        # 2. Skip high-cardinality columns that may be identifiers.
        if is_id_like(feature_series, id_cardinality_threshold):
            continue

        # 3. Build single-column preprocessing and modeling pipeline
        is_numeric = pd.api.types.is_numeric_dtype(feature_series)
        X = clean_df[[col]]

        if is_numeric:
            preprocessor = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
        else:
            preprocessor = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])

        pipeline = Pipeline([
            ("prep", preprocessor),
            ("classifier", LogisticRegression(solver="liblinear", random_state=42)),
        ])

        # 4. Stratified CV to ensure balanced targets across folds
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        try:
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv_strategy,
                scoring=scoring,
                error_score="raise",
            )
            mean_score = float(np.mean(scores))

            if mean_score >= threshold:
                detail.append({"column": col, "score": round(mean_score, 4)})

        except Exception as exc:
            errors.append({"column": col, "error": f"{type(exc).__name__}: {exc}"})

    # Sort flagged features descending by score
    detail = sorted(detail, key=lambda x: x["score"], reverse=True)

    return {
        "check": "single_feature_predictiveness",
        "n_flagged": len(detail),
        "detail": detail,
        "scoring": scoring,
        "errors": errors,
    }


def check_correlation(
    df: pd.DataFrame, target_col: str, threshold: float = 0.9
) -> "CheckResult":
    """Computes direct correlation (numeric) or Cramér's V (categorical)

    against the target to quickly catch obvious leakage without model training.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    clean_df = df.dropna(subset=[target_col]).copy()
    y = clean_df[target_col]
    detail = []

    for col in clean_df.columns:
        if col == target_col:
            continue

        series = clean_df[col]

        if series.nunique(dropna=True) <= 1:
            continue

        if pd.api.types.is_numeric_dtype(series):
            # Pearson correlation for numeric columns
            corr = abs(series.corr(y))
            score = float(corr) if not pd.isna(corr) else 0.0
        else:
            # Cramér's V for categorical columns
            score = _cramers_v(series.astype(str), y.astype(str))

        if score >= threshold:
            detail.append({"column": col, "score": round(score, 4)})

    detail = sorted(detail, key=lambda x: x["score"], reverse=True)

    return {
        "check": "correlation",
        "n_flagged": len(detail),
        "detail": detail,
    }


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Internal helper to calculate Cramér's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty or confusion_matrix.size == 0:
        return 0.0

    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0

    return float(np.sqrt(phi2corr / denom))


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "Data" / "processed" / "leaky" / "leaky_target_train.csv"
    df = pd.read_csv(data_path)

    print(f"Checking target leakage in {data_path}")
    print("--- Single Feature Predictiveness Check ---")
    pred_report = check_single_feature_predictiveness(
        df, target_col="Churn", threshold=0.9
    )
    print(pred_report)

    print("\n--- Direct Correlation / Association Check ---")
    corr_report = check_correlation(df, target_col="Churn", threshold=0.9)
    print(corr_report)


if __name__ == "__main__":
    main()
