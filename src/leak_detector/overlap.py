"""
overlap.py

Checks for train/test overlap: exact duplicate rows and near-duplicate
rows that would let a model "see" test data during training.
"""

import hashlib

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _row_hash(row: pd.Series) -> str:
    """Turn a row's values into a single fingerprint we can compare."""
    row_str = "|".join(str(v) for v in row.values)
    return hashlib.md5(row_str.encode()).hexdigest()


def check_exact_duplicates(
    train_df: pd.DataFrame, test_df: pd.DataFrame, ignore_cols=None
) -> dict:
    """
    Flags rows that appear, byte-for-byte, in both train and test.

    ignore_cols: columns to exclude from the fingerprint (e.g. a unique
    ID column, which would otherwise hide real duplicates since every
    row's ID is different even when everything else matches).
    """
    ignore_cols = ignore_cols or []
    cols = [c for c in train_df.columns if c not in ignore_cols]

    train_hashes = train_df[cols].apply(_row_hash, axis=1)
    test_hashes = test_df[cols].apply(_row_hash, axis=1)

    train_hash_to_idx = {}
    for idx, h in train_hashes.items():
        train_hash_to_idx.setdefault(h, []).append(idx)

    flagged = []
    for test_idx, h in test_hashes.items():
        if h in train_hash_to_idx:
            flagged.append({"test_idx": test_idx, "train_idx": train_hash_to_idx[h]})

    return {"check": "exact_overlap", "n_flagged": len(flagged), "detail": flagged}


def check_near_duplicates(
    train_df: pd.DataFrame, test_df: pd.DataFrame, threshold: float = 0.95
) -> dict:
    """
    Flags test rows whose nearest training-set neighbor is suspiciously
    close, after scaling numeric columns and one-hot encoding categoricals.
    """
    if len(train_df) == 0 or len(test_df) == 0:
        return {"check": "near_duplicate", "n_flagged": 0, "detail": []}

    numeric_cols = train_df.select_dtypes(include="number").columns.tolist()
    categorical_cols = train_df.select_dtypes(exclude="number").columns.tolist()

    transformer = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    train_encoded = transformer.fit_transform(train_df)
    test_encoded = transformer.transform(test_df)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(train_encoded)
    distances, indices = nn.kneighbors(test_encoded)

    flagged = []
    for i, (dist, train_pos) in enumerate(zip(distances[:, 0], indices[:, 0])):
        similarity = 1 / (1 + dist)
        if similarity >= threshold:
            flagged.append(
                {
                    "test_idx": test_df.index[i],
                    "train_idx": train_df.index[train_pos],
                    "similarity": round(float(similarity), 4),
                }
            )

    return {"check": "near_duplicate", "n_flagged": len(flagged), "detail": flagged}


if __name__ == "__main__":
    # Quick sanity check against the fixtures from prepare_fixtures.py.
    # Run from the folder that contains fixtures/.
    clean_train = pd.read_csv(r"D:\WORK\projects\Data_Leakage_DiagnosticTool\Data\leaky_data\leaky_overlap_test.csv")
    clean_test = pd.read_csv(r"D:\WORK\projects\Data_Leakage_DiagnosticTool\Data\leaky_data\leaky_overlap_test.csv")
    leaky_train = pd.read_csv(r"D:\WORK\projects\Data_Leakage_DiagnosticTool\Data\Original dataset\clean_test.csv")
    leaky_test = pd.read_csv(r"D:\WORK\projects\Data_Leakage_DiagnosticTool\Data\Original dataset\clean_test.csv")

    print("Near-duplicate check:")
    print("Clean data  :", check_near_duplicates(clean_train, clean_test))
    print("Leaky data  :", check_near_duplicates(leaky_train, leaky_test))
    print("Exact-duplicate check:")
    print("Clean data  :", check_exact_duplicates(clean_train, clean_test, ignore_cols=["customerID"]))
    print("Leaky data  :", check_exact_duplicates(leaky_train, leaky_test, ignore_cols=["customerID"]))

