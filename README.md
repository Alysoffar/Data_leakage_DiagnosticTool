# Data Leakage Diagnostic Tool

A focused Python tool for detecting common forms of data leakage in tabular machine-learning datasets. It compares a training CSV with a test CSV, runs overlap and target-leakage checks, and writes a Markdown report that can be reviewed or attached to an experiment record.

The project is designed for two workflows:

- Run the detector against an existing train/test split.
- Generate clean and intentionally contaminated datasets for validation and stress testing.

## Why leakage detection matters

Data leakage occurs when information from outside a model's allowed training boundary influences the training data or a feature. Leakage can make validation metrics look excellent while the model performs poorly on genuinely unseen data.

This tool is diagnostic, not a replacement for understanding how a dataset was collected. A clean report means that these checks did not find suspicious patterns; it does not prove that the entire data pipeline is leakage-free.

## Checks included

### Train/test overlap

The overlap checks compare rows between the training and test datasets.

- **Exact duplicates** identify test rows that also appear in training.
- **Near duplicates** use a similarity threshold to identify rows that are almost identical.
- Identifier columns can be excluded from exact-duplicate fingerprints with `--ignore-cols`.

Exact duplicates are usually a serious problem because the model may have already seen the test record during training. Near duplicates require investigation because they may be legitimate repeated observations or may indicate that the split was performed after related records were created.

### Single-feature target predictiveness

Each feature is evaluated by itself with preprocessing and cross-validated logistic regression. A feature whose standalone ROC AUC reaches the configured threshold may be a leaked representation or an unusually strong proxy for the target.

This check currently requires a binary target. A target with three or more classes raises a clear validation error instead of producing an invalid binary AUC result.

### Direct correlation and association

Numeric features are checked with absolute Pearson correlation. Categorical features are checked with Cramer's V. Features at or above the configured target threshold are included in the report.

These scores are screening signals, not causal conclusions. A high association should be investigated in the feature-generation process.

## Repository layout

```text
Data/
  raw/                         Original source CSV files
  processed/
    clean/                     Default clean train/test pair
    leaky/                     Default intentionally leaky pairs
    <dataset-name>/            Named dataset variants, such as adult/
      clean/
      leaky/
reports/                       Generated Markdown reports
src/
  Data_preprocessing/
    prepare_fixtures.py        Dataset split and leak-injection utility
  leak_detector/
    core.py                    CLI and reusable LeakageDiagnostic class
    overlap.py                 Exact and near-overlap checks
    paths.py                   Canonical project paths
    report.py                  Markdown report generation
    target_leakage.py          Predictiveness and association checks
examples/                      Notebook demonstrations
tests/                         Automated tests using Data/processed
```

`Data/` is the canonical dataset location. Tests do not maintain a second copy of the CSV fixtures. Generated reports are written under `reports/` and are ignored by Git.

## Requirements

- Python 3.10 or newer
- pandas 2.0 or newer
- NumPy 1.24 or newer
- scikit-learn 1.4 or newer
- SciPy, required by the categorical association check
- pytest, required for the test suite

The runtime dependencies are listed in `pyproject.toml`. A development environment should also install pytest from `requirements.txt` if it is listed there.

## Installation

From the repository root, create or activate a virtual environment and install the project in editable mode:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Editable installation makes the `leak-check` command available and ensures that changes under `src/` are used immediately.

## Prepare datasets

### Use an existing train/test pair

If your data is already split, place the source data under `Data/` or keep it at another path. The detector accepts any readable CSV path; the files do not have to be inside the repository.

### Generate the default Telco variants

Place the raw Telco CSV at `Data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`, then run:

```powershell
python src/Data_preprocessing/prepare_fixtures.py `
  Data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The default target column is `Churn`. The command writes:

```text
Data/processed/clean/clean_train.csv
Data/processed/clean/clean_test.csv
Data/processed/leaky/leaky_overlap_train.csv
Data/processed/leaky/leaky_overlap_test.csv
Data/processed/leaky/leaky_target_train.csv
Data/processed/leaky/leaky_target_test.csv
```

The generated leaky variants are useful for verifying that the detector catches known problems:

- `leaky_overlap_*` contains copied test rows in the training data.
- `leaky_target_*` contains a synthetic `cheat_score` derived from the target.

### Generate variants for another dataset

The preparation utility accepts any CSV with a binary target column. Give each dataset a name so its outputs do not overwrite another dataset:

```powershell
python src/Data_preprocessing/prepare_fixtures.py `
  Data/raw/adult.csv `
  --target-column income `
  --dataset-name adult
```

This creates:

```text
Data/processed/adult/clean/clean_train.csv
Data/processed/adult/clean/clean_test.csv
Data/processed/adult/leaky/leaky_overlap_train.csv
Data/processed/adult/leaky/leaky_overlap_test.csv
Data/processed/adult/leaky/leaky_target_train.csv
Data/processed/adult/leaky/leaky_target_test.csv
```

The target values are encoded as `0` and `1` during preparation so the synthetic target-leakage feature can be generated consistently. The original raw CSV is not modified.

## Run the detector

Run commands from the repository root whenever possible.

### Installed command

```powershell
leak-check `
  Data/processed/clean/clean_train.csv `
  Data/processed/clean/clean_test.csv `
  --target Churn `
  --ignore-cols customerID
```

For the Adult dataset:

```powershell
leak-check `
  Data/processed/adult/clean/clean_train.csv `
  Data/processed/adult/clean/clean_test.csv `
  --target income
```

### Run without the console entry point

```powershell
python src/leak_detector/core.py `
  Data/processed/clean/clean_train.csv `
  Data/processed/clean/clean_test.csv `
  --target Churn `
  --ignore-cols customerID
```

When no `--output` is supplied, the report is written to:

```text
reports/leak_report.md
```

A relative output name is also placed under `reports/`:

```powershell
leak-check train.csv test.csv --target target --output experiment_01.md
```

An absolute output path is respected as provided:

```powershell
leak-check train.csv test.csv --target target `
  --output D:\results\experiment_01.md
```

### Command-line options

| Option | Required | Default | Description |
|---|---:|---:|---|
| `train_path` | Yes | None | Training CSV path |
| `test_path` | Yes | None | Test CSV path |
| `--target` | Yes | None | Target column name |
| `--output` | No | `reports/leak_report.md` | Markdown report path |
| `--ignore-cols` | No | None | Columns excluded from exact duplicate fingerprints |
| `--overlap-threshold` | No | `0.95` | Similarity threshold for near duplicates |
| `--target-threshold` | No | `0.90` | AUC or association threshold for target leakage |

`--ignore-cols` accepts zero or more column names after the option:

```powershell
leak-check train.csv test.csv --target Churn --ignore-cols customerID account_id
```

## Read the report

Each report contains:

1. An overall verdict.
2. The number of rows in the test set.
3. Exact and near-duplicate counts.
4. A sample of affected rows and similarity details.
5. Single-feature predictiveness results.
6. Direct correlation or association results.
7. The target column and configured thresholds.

A clean report is not a guarantee of clean data. Review the report together with the feature lineage, collection timestamps, entity boundaries, and any preprocessing performed before the split.

## ID detection behavior

The predictiveness check uses a heuristic named `is_id_like` to avoid treating identifiers as useful predictive features.

The current heuristic skips:

- High-cardinality text columns.
- High-cardinality categorical columns.
- High-cardinality monotonic integer keys.

It does not automatically skip every high-cardinality numeric column. Continuous measurements can legitimately have many unique values, and a noisy target proxy must remain visible to the leakage check. Because this is a heuristic, unusual identifiers should be tested explicitly and reviewed with domain knowledge.

If an identifier is not detected automatically, pass it to `--ignore-cols` for overlap detection and consider excluding it from modeling separately.

## Input validation and errors

The CLI reports clear errors for:

- Missing training or test files.
- Empty files or CSVs with no data rows.
- Malformed CSV input.
- A target column missing from either split.
- A non-binary target used by the single-feature predictiveness check.

Example:

```text
leak-check: error: Training CSV file not found: missing_train.csv
```

Always use matching datasets for the two positional arguments. Do not compare a Telco training file with an Adult test file, even if both files are valid CSVs. The columns, target, entities, and data-generating process must belong to the same dataset and split.

If your shell is already inside `src/leak_detector`, run `python core.py ...` or return to the repository root. Do not prefix the command with `src/leak_detector/` from that subdirectory.

## Python API

The existing function API is available for scripts and notebooks:

```python
from leak_detector import run_leak_check

summary = run_leak_check(
    train_path="Data/processed/adult/clean/clean_train.csv",
    test_path="Data/processed/adult/clean/clean_test.csv",
    target_col="income",
    ignore_cols=["record_number"],
    output_path="adult_report.md",
)

print(summary)
```

For reusable configuration, use `LeakageDiagnostic`:

```python
from leak_detector import LeakageDiagnostic

checker = LeakageDiagnostic(
    target_col="income",
    ignore_cols=["record_number"],
    overlap_threshold=0.95,
    target_threshold=0.90,
)

summary = checker.run(
    "Data/processed/adult/clean/clean_train.csv",
    "Data/processed/adult/clean/clean_test.csv",
)
```

The returned summary contains the counts and report path:

```python
{
    "exact_overlap_flagged": 0,
    "near_overlap_flagged": 0,
    "target_predictiveness_flagged": 0,
    "target_correlation_flagged": 0,
    "report_path": ".../reports/leak_report.md",
}
```

## Testing

Run the complete suite from the repository root:

```powershell
pytest -q
```

The tests cover:

- Exact and near train/test overlap.
- Intentionally injected target leakage.
- Binary-target validation.
- Alternate numeric identifier shapes.
- Missing files, empty CSVs, and missing target columns.
- End-to-end Markdown report generation.

## Development notes

The project uses a `src/` layout and is configured through `pyproject.toml`. Keep reusable library code under `src/leak_detector/`, data-generation utilities under `src/Data_preprocessing/`, and tests under `tests/`.

Generated Markdown reports belong in `reports/`. Keep raw inputs and processed datasets organized under `Data/`, and use a named processed subdirectory for every additional dataset.

## Limitations

This tool is intentionally conservative and diagnostic:

- It does not inspect SQL queries, feature-store lineage, or upstream transformation code.
- It does not detect every temporal or entity-level leakage pattern.
- Near-duplicate similarity depends on how mixed numeric and categorical columns are represented.
- Correlation and AUC thresholds are screening choices, not universal correctness criteria.
- The single-feature predictiveness check currently supports binary targets only.

Use the results to guide investigation of the data split and feature-generation process rather than treating the report as a formal proof of data quality.

## License

See [LICENSE](LICENSE) for the project license.
