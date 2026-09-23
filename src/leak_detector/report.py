"""
report.py

Turns the raw output of overlap.py and target_leakage.py's checkers
into a single human-readable Markdown report: what was checked, what
it means, and the actual rows/columns that got flagged.
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .types import CheckResult


def _rows_to_markdown_table(df: pd.DataFrame, indices, max_cols=8, max_rows=15) -> str:
    """
    Renders a subset of a dataframe's rows as a Markdown table.
    Truncates wide dataframes (max_cols) and long lists (max_rows) so
    the report stays readable instead of dumping the whole dataset.
    """
    if len(indices) == 0:
        return "_None found._\n"

    shown_indices = indices[:max_rows]
    subset = df.loc[shown_indices]

    cols = list(subset.columns)
    truncated_cols = len(cols) > max_cols
    if truncated_cols:
        cols = cols[:max_cols]
    subset = subset[cols]

    header = "| index | " + " | ".join(cols) + (" | ... |" if truncated_cols else " |")
    separator = "|---" * (len(cols) + 1 + (1 if truncated_cols else 0)) + "|"

    lines = [header, separator]
    for idx, row in subset.iterrows():
        values = " | ".join(str(v) for v in row.values)
        line = f"| {idx} | {values} |"
        if truncated_cols:
            line = line[:-1] + " ... |"
        lines.append(line)

    table = "\n".join(lines)
    if len(indices) > max_rows:
        table += f"\n\n_...and {len(indices) - max_rows} more, not shown._\n"

    return table


def _build_overlap_section(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exact_result: "CheckResult",
    near_result: "CheckResult",
    group_result: "CheckResult | None" = None,
) -> list:
    """Builds the overlap section as a list of Markdown lines (no file I/O)."""
    group_result = group_result or {"group_col": None, "n_flagged": 0, "detail": []}
    exact_flags = exact_result["detail"]
    near_flags = near_result["detail"]

    exact_test_idx = [f["test_idx"] for f in exact_flags]
    near_test_idx = [f["test_idx"] for f in near_flags]
    group_test_idx = group_result.get("test_indices", [])

    total_flagged = len(set(exact_test_idx) | set(near_test_idx) | set(group_test_idx))
    pct_of_test = 100 * total_flagged / len(test_df) if len(test_df) else 0

    if (
        exact_result["n_flagged"] == 0
        and near_result["n_flagged"] == 0
        and group_result["n_flagged"] == 0
    ):
        verdict = "**No train/test overlap detected.** This split looks safe to trust."
    elif exact_result["n_flagged"] > 0:
        verdict = (
            "**Exact duplicate rows found between train and test.** "
            "This is a serious leak — the model is being tested on data "
            "it already memorized during training. Fix before trusting any metric."
        )
    elif near_result["n_flagged"] > 0:
        verdict = (
            "**Near-duplicate rows found between train and test.** "
            "Not exact copies, but close enough that the model may be "
            "getting an easier test than it will face in production. Worth investigating."
        )
    else:
        verdict = (
            "**Shared entity groups found between train and test.** "
            "Related observations cross the split boundary, so test performance "
            "may be optimistic. Investigate the split by entity."
        )

    lines = ["## Train/Test Overlap"]
    lines.append(
        "If the same (or nearly the same) rows show up in both your training "
        "and test data, your test score isn't measuring how well the model "
        "generalizes — it's measuring how well it memorized. This checks for:\n"
        "- **Exact duplicates** — identical rows in both sets\n"
        "- **Near duplicates** — rows that are almost, but not exactly, the same\n"
    )
    lines.append(f"- Test set size: **{len(test_df)}** rows")
    lines.append(f"- Exact duplicates flagged: **{exact_result['n_flagged']}**")
    lines.append(f"- Near duplicates flagged: **{near_result['n_flagged']}**")
    group_label = (
        f" using **{group_result['group_col']}**"
        if group_result.get("group_col")
        else " (not run: no group column)"
    )
    lines.append(
        f"- Shared groups flagged: **{group_result['n_flagged']}** test rows{group_label}"
    )
    lines.append(f"- Total unique test rows affected: **{total_flagged}** ({pct_of_test:.1f}% of test set)\n")
    lines.append(f"**Verdict:** {verdict}\n")

    lines.append("### Exact duplicate rows")
    lines.append("These test rows are byte-for-byte identical to a row somewhere in training.\n")
    lines.append(_rows_to_markdown_table(test_df, exact_test_idx))
    lines.append("")

    lines.append("### Near-duplicate rows")
    lines.append(
        f"These test rows matched a training row with similarity >= threshold "
        f"({near_result.get('threshold', 'see near_result')}). Similarity score shown per row.\n"
    )
    if near_flags:
        sim_lines = [
            f"- index `{f['test_idx']}` <-> train index `{f['train_idx']}` -- similarity **{f['similarity']}**"
            for f in near_flags[:15]
        ]
        if len(near_flags) > 15:
            sim_lines.append(f"- ...and {len(near_flags) - 15} more, not shown.")
        lines.append("\n".join(sim_lines))
        lines.append("")
        lines.append(_rows_to_markdown_table(test_df, near_test_idx))
    else:
        lines.append("_None found._")

    lines.append("")
    lines.append("### Shared entity groups")
    if group_result.get("group_col"):
        lines.append(
            f"Entities in **{group_result['group_col']}** that have observations in both splits:"
        )
        group_flags = group_result["detail"]
        if group_flags:
            lines.extend(
                f"- **{flag['group']}** -- train rows: **{flag['train_count']}**, "
                f"test rows: **{flag['test_count']}**"
                for flag in group_flags[:15]
            )
            if len(group_flags) > 15:
                lines.append(f"- ...and {len(group_flags) - 15} more, not shown.")
        else:
            lines.append("_None found._")
    else:
        lines.append("_Not run because neither input contains a recognized entity ID column._")

    return lines


def _build_target_leakage_section(
    df: pd.DataFrame,
    target_col: str,
    predictiveness_result: "CheckResult",
    correlation_result: "CheckResult",
) -> list:
    """Builds the target-leakage section as a list of Markdown lines (no file I/O)."""
    predictiveness_flags = predictiveness_result["detail"]
    correlation_flags = correlation_result["detail"]
    total_flagged = len(
        {flag["column"] for flag in predictiveness_flags}
        | {flag["column"] for flag in correlation_flags}
    )

    if total_flagged == 0:
        verdict = "**No likely target leakage detected.**"
    else:
        verdict = (
            f"**{total_flagged} feature(s) may contain target leakage.** "
            "Investigate these features before trusting model results."
        )

    lines = [
        "## Target Leakage",
        (
            "This checks whether any single column can predict the target "
            "suspiciously well on its own -- a sign it's a leaked or proxy "
            "version of the answer rather than a genuine feature.\n"
        ),
        f"- Dataset rows: **{len(df)}**",
        f"- Target column: **{target_col}**",
        f"- Predictive features flagged: **{predictiveness_result['n_flagged']}**",
        f"- Highly correlated features flagged: **{correlation_result['n_flagged']}**",
        f"- Total unique features flagged: **{total_flagged}**\n",
        f"**Verdict:** {verdict}\n",
        "### Single-feature predictiveness",
        "Features with a cross-validated AUC at or above the configured threshold:",
    ]

    if predictiveness_result.get("errors"):
        lines.append(
            f"- Features skipped because evaluation failed: **{len(predictiveness_result['errors'])}**"
        )
        lines.extend(
            f"  - `{error['column']}`: {error['error']}"
            for error in predictiveness_result["errors"][:15]
        )

    if predictiveness_flags:
        lines.extend(f"- **{f['column']}** -- AUC: **{f['score']}**" for f in predictiveness_flags)
    else:
        lines.append("_None found._")

    lines.extend(
        [
            "\n### Direct correlation / association",
            "Features with a correlation or Cramer's V score at or above the configured threshold:",
        ]
    )
    if correlation_flags:
        lines.extend(f"- **{f['column']}** -- score: **{f['score']}**" for f in correlation_flags)
    else:
        lines.append("_None found._")

    return lines


def _build_temporal_section(
    feature_timestamp_result: "CheckResult",
    split_chronology_result: "CheckResult",
) -> list:
    """Build the temporal leakage section without performing file I/O."""
    feature_flags = feature_timestamp_result["detail"]
    split_detail = split_chronology_result["detail"]
    temporal_run = bool(
        feature_timestamp_result.get("label_date_col")
        or split_detail.get("date_col")
        or feature_flags
        or split_chronology_result["n_flagged"]
    )
    if not temporal_run:
        return []

    lines = [
        "## Temporal Leakage",
        "This checks whether feature timestamps occur after the label date and whether training rows extend into the test period.",
        f"- Feature timestamp violations: **{feature_timestamp_result['n_flagged']}**",
        f"- Training rows after the earliest test date: **{split_chronology_result['n_flagged']}**",
        f"- Train maximum date: **{split_detail.get('train_max_date')}**",
        f"- Test minimum date: **{split_detail.get('test_min_date')}**",
        f"- Chronological split: **{split_detail.get('is_chronological')}**\n",
    ]
    if feature_flags:
        lines.append("### Feature timestamp violations")
        lines.extend(
            f"- row `{flag['row_idx']}` / **{flag['column']}**: "
            f"label `{flag['label_date']}`; feature `{flag['feature_date']}`"
            for flag in feature_flags[:15]
        )
        if len(feature_flags) > 15:
            lines.append(f"- ...and {len(feature_flags) - 15} more, not shown.")
    else:
        lines.extend(["### Feature timestamp violations", "_None found._"])

    lines.extend(["", "### Split chronology"])
    if split_chronology_result["n_flagged"]:
        lines.append("**Training contains rows later than the earliest test row.**")
    else:
        lines.append("_No chronology violation found._")
    return lines


def _build_code_section(code_result: "CheckResult") -> list:
    """Build the optional static preprocessing leakage section."""
    if code_result.get("skipped"):
        return []

    lines = [
        "## Static Code Leakage",
        "This scans Python source for preprocessing performed before splitting or outside a cross-validation fold.",
        f"- Violations flagged: **{code_result['n_flagged']}**",
    ]
    if code_result.get("error"):
        lines.append(f"- Parse error: `{code_result['error']}`")
    if code_result["detail"]:
        lines.append("")
        lines.extend(
            f"- line **{violation['line']}** / `{violation['method']}`: {violation['issue']}"
            for violation in code_result["detail"][:15]
        )
        if len(code_result["detail"]) > 15:
            lines.append(f"- ...and {len(code_result['detail']) - 15} more, not shown.")
    else:
        lines.append("_No static preprocessing leakage found._")
    return lines


def generate_full_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    exact_result: "CheckResult",
    near_result: "CheckResult",
    predictiveness_result: "CheckResult",
    correlation_result: "CheckResult",
    output_path: str | Path = "leak_report.md",
    group_result: "CheckResult | None" = None,
    feature_timestamp_result: "CheckResult | None" = None,
    split_chronology_result: "CheckResult | None" = None,
    code_result: "CheckResult | None" = None,
    metadata: dict | None = None,
) -> str:
    """
    Builds ONE combined Markdown report covering both overlap and target
    leakage checks, and writes it to output_path.
    """
    group_result = group_result or {"group_col": None, "n_flagged": 0, "detail": []}
    feature_timestamp_result = feature_timestamp_result or {
        "n_flagged": 0,
        "detail": [],
    }
    split_chronology_result = split_chronology_result or {
        "n_flagged": 0,
        "detail": {
            "train_max_date": None,
            "test_min_date": None,
            "is_chronological": True,
        },
    }
    code_result = code_result or {
        "check": "static_code_leakage",
        "n_flagged": 0,
        "detail": [],
        "skipped": True,
    }
    metadata = metadata or {}
    overlap_clean = (
        exact_result["n_flagged"] == 0
        and near_result["n_flagged"] == 0
        and group_result["n_flagged"] == 0
    )
    target_clean = predictiveness_result["n_flagged"] == 0 and correlation_result["n_flagged"] == 0
    temporal_clean = (
        feature_timestamp_result["n_flagged"] == 0
        and split_chronology_result["n_flagged"] == 0
    )
    code_clean = code_result.get("skipped", False) or code_result["n_flagged"] == 0

    if overlap_clean and target_clean and temporal_clean and code_clean:
        overall = "No leakage detected by either check."
    else:
        categories = []
        if not overlap_clean:
            categories.append("train/test overlap")
        if not target_clean:
            categories.append("target leakage")
        if not temporal_clean:
            categories.append("temporal leakage")
        if not code_clean:
            categories.append("static code leakage")
        overall = "Leakage detected: " + ", ".join(categories) + "."

    lines = [
        "# Data Leakage Report",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n",
        f"_Schema: `1.0` | Profile: `{metadata.get('profile', 'full')}` | Target: `{target_col}`_\n",
        f"## Overall verdict\n{overall}\n",
        "---\n",
    ]
    lines += _build_overlap_section(train_df, test_df, exact_result, near_result, group_result)
    lines.append("\n---\n")
    lines += _build_target_leakage_section(train_df, target_col, predictiveness_result, correlation_result)
    temporal_lines = _build_temporal_section(feature_timestamp_result, split_chronology_result)
    if temporal_lines:
        lines.append("\n---\n")
        lines += temporal_lines
    code_lines = _build_code_section(code_result)
    if code_lines:
        lines.append("\n---\n")
        lines += code_lines

    report_text = "\n".join(lines)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


if __name__ == "__main__":
    from overlap import check_exact_duplicates, check_near_duplicates
    from paths import ProjectPaths
    from target_leakage import check_correlation, check_single_feature_predictiveness

    paths = ProjectPaths.from_file(__file__)
    target_col = "Churn"

    # Step 7 validation: run BOTH checkers against ALL THREE fixture pairs,
    # not just the pair each checker was designed to catch. This is what
    # actually proves the tool works end-to-end.
    runs = {
        "clean": ("clean_train.csv", "clean_test.csv"),
        "leaky_overlap": ("leaky_overlap_train.csv", "leaky_overlap_test.csv"),
        "leaky_target": ("leaky_target_train.csv", "leaky_target_test.csv"),
    }

    for label, (train_file, test_file) in runs.items():
        if label == "clean":
            fixtures_dir = paths.clean_data
        else:
            fixtures_dir = paths.leaky_data

        train = pd.read_csv(fixtures_dir / train_file)
        test = pd.read_csv(fixtures_dir / test_file)

        exact_result = check_exact_duplicates(train, test, ignore_cols=["customerID"])
        near_result = check_near_duplicates(train, test, threshold=0.95)
        predictiveness_result = check_single_feature_predictiveness(train, target_col=target_col, threshold=0.9)
        correlation_result = check_correlation(train, target_col=target_col, threshold=0.9)

        out_path = paths.report_path(f"{label}_report.md")
        generate_full_report(
            train, test, target_col,
            exact_result, near_result,
            predictiveness_result, correlation_result,
            out_path,
        )

        print(
            f"[{label}] exact={exact_result['n_flagged']} "
            f"near={near_result['n_flagged']} "
            f"predictive={predictiveness_result['n_flagged']} "
            f"correlation={correlation_result['n_flagged']} "
            f"-> {out_path}"
        )