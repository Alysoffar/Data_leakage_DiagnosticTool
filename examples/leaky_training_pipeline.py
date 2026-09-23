"""Intentionally leaky training script for the static pipeline checker.

This file is a test fixture, not production training code. It contains both
unsafe and safe patterns so the checker can be exercised on a realistic script.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    return data.drop(columns=["Churn"]), data["Churn"]


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    numeric = data.select_dtypes(include="number").columns
    transformed = data.copy()
    # LEAK 1: statistics are learned from the full dataset before splitting.
    transformed[numeric] = StandardScaler().fit_transform(transformed[numeric])
    return transformed


def unsafe_global_preprocessing(X: pd.DataFrame, y: pd.Series) -> None:
    X_processed = build_features(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # LEAK 2: the original unsplit data is fitted after the split.
    scaler = StandardScaler()
    scaler.fit(X)

    # LEAK 3: raw estimator receives globally transformed data in CV.
    model = LogisticRegression(max_iter=500)
    scores = cross_val_score(model, X_processed, y, cv=5, scoring="roc_auc")
    print(scores.mean(), len(X_train), len(X_test), y_train.mean(), y_test.mean())


def unsafe_manual_cross_validation(X: pd.DataFrame, y: pd.Series) -> None:
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kfold.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        # LEAK 4: this transformer sees validation rows inside every fold.
        fold_scaler = StandardScaler()
        fold_scaler.fit_transform(X)

        # SAFE: this model is fit only on the fold training slice.
        model = RandomForestClassifier(n_estimators=25, random_state=42)
        model.fit(X_train, y.iloc[train_idx])
        print(model.score(X_test, y.iloc[test_idx]))


def safe_pipeline_cross_validation(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    numeric = X.select_dtypes(include="number").columns
    preprocessing = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), numeric)],
        remainder="drop",
    )
    pipeline = Pipeline(
        [
            ("preprocess", preprocessing),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    # SAFE: preprocessing is fitted independently inside each CV fold.
    return cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")


if __name__ == "__main__":
    features, target = load_data("Data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    unsafe_global_preprocessing(features, target)
    unsafe_manual_cross_validation(features, target)
    print(safe_pipeline_cross_validation(features, target).mean())