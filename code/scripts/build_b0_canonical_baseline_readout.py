#!/usr/bin/env python3
"""Build the B0 canonical-evidence baseline readout memo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "t4_followup_20260505"
    / "metrics_stage2_plus_t4_plus_b0_four_models_20260510"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "docs"
    / "experiment_results"
    / "tables"
    / "b0_canonical_baseline_20260510"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "experiment_results"
    / "b0_canonical_baseline_readout_20260510.md"
)

TREATMENT_ORDER = ["T1", "B0", "T2", "T3", "T4"]
CONTRAST_ORDER = [
    "B0_minus_T1",
    "T2_minus_B0",
    "T3_minus_B0",
    "T4_minus_B0",
    "T3_minus_T2",
    "T4_minus_T2",
]
PRIMARY_METRICS = [
    ("mean_pairwise_abs_diff_expected_return_5d", "Belief MAD"),
    ("mean_pairwise_sentence_embedding_distance", "Rationale distance"),
    ("source_id_diversity_one_minus_overlap", "Source diversity"),
    ("evidence_category_diversity_one_minus_overlap", "Category diversity"),
    ("entropy_buy_hold_sell", "Action entropy"),
    ("action_diversity_one_minus_hhi", "Action diversity"),
]
QUALITY_METRICS = [
    ("quality_directional_accuracy", "Directional"),
    ("quality_action_accuracy", "Strict action"),
    ("quality_mean_absolute_return_error", "Abs err"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default=str(DEFAULT_METRICS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    return parser.parse_args()


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def fmt(value: Any, digits: int = 5) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def fmt_delta(value: Any, digits: int = 5) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.{digits}f}"


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join([header, sep, *body])


def ordered_categories(series: pd.Series, order: list[str]) -> pd.Series:
    return pd.Categorical(series, categories=order, ordered=True)


def treatment_means(cell: pd.DataFrame) -> pd.DataFrame:
    df = cell[cell["investor_group"].eq("all_profiles")].copy()
    cols = [key for key, _ in PRIMARY_METRICS] + [key for key, _ in QUALITY_METRICS]
    grouped = df.groupby("treatment_family", observed=True)[cols].mean().reset_index()
    grouped["treatment_family"] = ordered_categories(grouped["treatment_family"], TREATMENT_ORDER)
    return grouped.sort_values("treatment_family")


def treatment_contrasts(contrasts: pd.DataFrame) -> pd.DataFrame:
    wanted = {key: label for key, label in PRIMARY_METRICS}
    df = contrasts[
        contrasts["investor_group"].eq("all_profiles")
        & contrasts["contrast_id"].isin(CONTRAST_ORDER)
        & contrasts["metric_name"].isin(wanted)
    ].copy()
    grouped = (
        df.groupby(["contrast_id", "metric_name"], observed=True)["delta"]
        .mean()
        .reset_index()
    )
    wide = grouped.pivot(index="contrast_id", columns="metric_name", values="delta").reset_index()
    wide["contrast_id"] = ordered_categories(wide["contrast_id"], CONTRAST_ORDER)
    return wide.sort_values("contrast_id")


def profile_means(cell: pd.DataFrame) -> pd.DataFrame:
    df = cell[cell["investor_group"].isin(["retail", "institutional"])].copy()
    cols = [key for key, _ in PRIMARY_METRICS]
    grouped = (
        df.groupby(["investor_group", "treatment_family"], observed=True)[cols]
        .mean()
        .reset_index()
    )
    grouped["treatment_family"] = ordered_categories(grouped["treatment_family"], TREATMENT_ORDER)
    return grouped.sort_values(["investor_group", "treatment_family"])


def receiver_model_means(cell: pd.DataFrame) -> pd.DataFrame:
    df = cell[cell["investor_group"].eq("all_profiles")].copy()
    cols = [key for key, _ in PRIMARY_METRICS]
    grouped = (
        df.groupby(["model_family", "treatment_family"], observed=True)[cols]
        .mean()
        .reset_index()
    )
    grouped["treatment_family"] = ordered_categories(grouped["treatment_family"], TREATMENT_ORDER)
    return grouped.sort_values(["model_family", "treatment_family"])


def unpaired_mean_delta(means: pd.DataFrame, left: str, right: str, metric: str) -> float:
    values = means.set_index("treatment_family")
    return float(values.loc[left, metric] - values.loc[right, metric])


def table_rows_from_means(df: pd.DataFrame, treatment_col: str = "treatment_family") -> list[dict[str, str]]:
    rows = []
    for _, row in df.iterrows():
        item = {"Treatment": str(row[treatment_col])}
        for key, label in PRIMARY_METRICS:
            item[label] = fmt(row[key])
        for key, label in QUALITY_METRICS:
            if key in row:
                item[label] = fmt(row[key])
        rows.append(item)
    return rows


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def build_report(metrics_dir: Path, output_dir: Path, report_path: Path) -> None:
    cell = pd.read_csv(metrics_dir / "cell_metrics.csv", low_memory=False)
    contrasts = pd.read_csv(metrics_dir / "treatment_contrasts.csv", low_memory=False)
    run_summary = json.loads((metrics_dir / "run_summary.json").read_text(encoding="utf-8"))

    means = treatment_means(cell)
    contrast_means = treatment_contrasts(contrasts)
    profiles = profile_means(cell)
    models = receiver_model_means(cell)

    write_csv(output_dir / "b0_downstream_treatment_means_20260510.csv", means)
    write_csv(output_dir / "b0_treatment_contrasts_20260510.csv", contrast_means)
    write_csv(output_dir / "b0_profile_separability_20260510.csv", profiles)
    write_csv(output_dir / "b0_by_receiver_model_20260510.csv", models)

    treatment_rows = table_rows_from_means(means)
    treatment_columns = [("Treatment", "Treatment")] + [
        (label, label) for _, label in PRIMARY_METRICS + QUALITY_METRICS
    ]

    contrast_rows = []
    for _, row in contrast_means.iterrows():
        item = {"Contrast": str(row["contrast_id"])}
        for key, label in PRIMARY_METRICS:
            item[label] = fmt_delta(row.get(key))
        contrast_rows.append(item)
    contrast_columns = [("Contrast", "Contrast")] + [(label, label) for _, label in PRIMARY_METRICS]

    profile_rows = []
    for _, row in profiles.iterrows():
        item = {"Group": str(row["investor_group"]), "Treatment": str(row["treatment_family"])}
        for key, label in PRIMARY_METRICS:
            item[label] = fmt(row[key])
        profile_rows.append(item)
    profile_columns = [("Group", "Group"), ("Treatment", "Treatment")] + [
        (label, label) for _, label in PRIMARY_METRICS
    ]

    model_rows = []
    for _, row in models.iterrows():
        item = {"Receiver": str(row["model_family"]), "Treatment": str(row["treatment_family"])}
        for key, label in PRIMARY_METRICS:
            item[label] = fmt(row[key])
        model_rows.append(item)
    model_columns = [("Receiver", "Receiver"), ("Treatment", "Treatment")] + [
        (label, label) for _, label in PRIMARY_METRICS
    ]

    source = "source_id_diversity_one_minus_overlap"
    category = "evidence_category_diversity_one_minus_overlap"
    rationale = "mean_pairwise_sentence_embedding_distance"
    action = "entropy_buy_hold_sell"

    b0_minus_t1_source = contrast_means.loc[
        contrast_means["contrast_id"].eq("B0_minus_T1"), source
    ].iloc[0]
    b0_minus_t1_category = contrast_means.loc[
        contrast_means["contrast_id"].eq("B0_minus_T1"), category
    ].iloc[0]
    t4_minus_b0_source = contrast_means.loc[
        contrast_means["contrast_id"].eq("T4_minus_B0"), source
    ].iloc[0]
    t4_minus_b0_category = contrast_means.loc[
        contrast_means["contrast_id"].eq("T4_minus_B0"), category
    ].iloc[0]
    t4_minus_b0_rationale = contrast_means.loc[
        contrast_means["contrast_id"].eq("T4_minus_B0"), rationale
    ].iloc[0]
    t4_minus_t2_source = contrast_means.loc[
        contrast_means["contrast_id"].eq("T4_minus_T2"), source
    ].iloc[0]
    t4_minus_t2_category = contrast_means.loc[
        contrast_means["contrast_id"].eq("T4_minus_T2"), category
    ].iloc[0]
    t4_minus_t2_rationale = contrast_means.loc[
        contrast_means["contrast_id"].eq("T4_minus_T2"), rationale
    ].iloc[0]
    t4_minus_t2_action = contrast_means.loc[
        contrast_means["contrast_id"].eq("T4_minus_T2"), action
    ].iloc[0]

    mean_deltas = [
        {
            "Contrast": "T4 - T3",
            "Source diversity": fmt_delta(unpaired_mean_delta(means, "T4", "T3", source)),
            "Category diversity": fmt_delta(unpaired_mean_delta(means, "T4", "T3", category)),
            "Rationale distance": fmt_delta(unpaired_mean_delta(means, "T4", "T3", rationale)),
            "Action entropy": fmt_delta(unpaired_mean_delta(means, "T4", "T3", action)),
            "Basis": "unpaired treatment means",
        },
        {
            "Contrast": "T4 - B0",
            "Source diversity": fmt_delta(t4_minus_b0_source),
            "Category diversity": fmt_delta(t4_minus_b0_category),
            "Rationale distance": fmt_delta(t4_minus_b0_rationale),
            "Action entropy": fmt_delta(
                contrast_means.loc[contrast_means["contrast_id"].eq("T4_minus_B0"), action].iloc[0]
            ),
            "Basis": "paired contrast",
        },
        {
            "Contrast": "T4 - T2",
            "Source diversity": fmt_delta(t4_minus_t2_source),
            "Category diversity": fmt_delta(t4_minus_t2_category),
            "Rationale distance": fmt_delta(t4_minus_t2_rationale),
            "Action entropy": fmt_delta(t4_minus_t2_action),
            "Basis": "paired contrast",
        },
    ]

    summary = run_summary["summary"]
    lines = [
        "# B0 Canonical Evidence Baseline Readout - 2026-05-10",
        "",
        "## Scope",
        "",
        "This memo adds the B0 canonical-evidence-only baseline to the existing",
        "T1/T2/T3/T4 bottleneck analysis frame. B0 directly sends the canonical",
        "evidence-unit bank to downstream synthetic investor decision-makers. It",
        "does not use an LLM-generated summary, the T4 pipe-table ledger, category",
        "regrouping, source-quote expansion, or regex reconstruction.",
        "",
        "Completed evidence:",
        "",
        "- Dataset: `94` clean strict earnings events, balanced `47` large-cap and `47` small/mid-cap.",
        "- B0 downstream: `4` receiver model engines x `94` events x `6` profiles = `2256` validated decision rows.",
        "- Full T1/T2/T3/T4+B0 merged decision file: "
        f"`{summary['total_input_rows']}` rows, `{summary['usable_decisions']}` usable decisions, `{summary['warnings']}` metric warnings.",
        "- B0-inclusive metrics: "
        f"`{summary['cell_metric_rows_written']}` cell rows, `{summary['t2_t3_contrast_rows_written']}` T2/T3 contrast rows, "
        f"`{summary['treatment_contrast_rows_written']}` long treatment-contrast rows.",
        "",
        "## Bottom Line",
        "",
        "B0 confirms that canonical-evidence exposure restores source/category",
        "multiplicity without proportionally increasing expressed rationale",
        "diversity or action dispersion. This separates citation/source-path",
        "diversity from reasoning/action diversity.",
        "",
        f"- `B0 - T1`: source `{fmt_delta(b0_minus_t1_source)}`, category `{fmt_delta(b0_minus_t1_category)}`.",
        f"- `T4 - B0`: source `{fmt_delta(t4_minus_b0_source)}`, category `{fmt_delta(t4_minus_b0_category)}`, rationale `{fmt_delta(t4_minus_b0_rationale)}`.",
        f"- `T4 - T2`: source `{fmt_delta(t4_minus_t2_source)}`, category `{fmt_delta(t4_minus_t2_category)}`, "
        f"rationale `{fmt_delta(t4_minus_t2_rationale)}`, action entropy `{fmt_delta(t4_minus_t2_action)}`.",
        "",
        "Interpretation: T4's high source/category diversity is not merely a T4",
        "ledger artifact. B0, which removes the T4 ledger and uses the canonical",
        "evidence bank directly, produces similarly high source/category diversity.",
        "The remaining paradox is that these diversified cited evidence paths do not",
        "translate into equally diversified rationales or actions.",
        "",
        "## Downstream Treatment Means",
        "",
        md_table(treatment_rows, treatment_columns),
        "",
        "Readout: B0 is close to T4 on source/category diversity and close to T1/T4",
        "on rationale diversity. T2/T3 have lower source/category diversity but",
        "higher action entropy, with T3 retaining the highest rationale distance.",
        "",
        "## B0-Inclusive Treatment Context",
        "",
        "Positive deltas mean the comparison treatment is higher than the baseline.",
        "",
        md_table(contrast_rows, contrast_columns),
        "",
        "B0 is the clean canonical substrate baseline. `T2 - B0` and `T3 - B0`",
        "therefore isolate the effect of adding LLM narrative synthesis on top of",
        "the canonical evidence bank. `T4 - B0` isolates the incremental effect of",
        "the T4 structured-ledger interface relative to the same canonical substrate.",
        "",
        "## B0 and the T4 Paradox",
        "",
        md_table(
            mean_deltas,
            [
                ("Contrast", "Contrast"),
                ("Source diversity", "Source"),
                ("Category diversity", "Category"),
                ("Rationale distance", "Rationale"),
                ("Action entropy", "Action"),
                ("Basis", "Basis"),
            ],
        ),
        "",
        "This makes the T4 paradox more precise. T4 is not simply the result of",
        "showing more complete information. Canonical evidence exposure already",
        "produces high source/category diversity in B0. The important result is the",
        "decoupling: more diverse cited evidence paths do not necessarily become",
        "more diverse stated rationales or more dispersed actions.",
        "",
        "## Profile Separability",
        "",
        md_table(profile_rows, profile_columns),
        "",
        "Profile readout: B0 and T4 both raise source/category separability for",
        "retail and institutional profiles. Their rationale separability remains",
        "well below T3, so the source-path expansion is not transmitted cleanly into",
        "expressed reasoning.",
        "",
        "## Receiver-Model Readout",
        "",
        md_table(model_rows, model_columns),
        "",
        "Model readout: the qualitative B0/T4 pattern is not limited to one receiver",
        "engine. Across the four model families, B0 and T4 are the source/category",
        "high-diversity conditions, while T3 remains the strongest rationale-distance",
        "condition.",
        "",
        "## Paper-Facing Interpretation",
        "",
        "B0 supports a cleaner mechanism claim:",
        "",
        "> Canonical evidence exposure increases source-path diversity, but source-path",
        "> diversity is not automatically transmitted into rationale or action",
        "> diversity. LLM synthesis and structured interfaces therefore operate at",
        "> different layers of the decision pipeline.",
        "",
        "The paper should avoid saying that T4 alone proves that full-evidence",
        "interfaces diversify source selection. B0 is the cleaner evidence for that",
        "point. T4 is now better interpreted as a structured-ledger boundary condition",
        "that largely preserves the B0 source/category pattern while failing to",
        "restore T3-level rationale diversity.",
        "",
        "Generated tables:",
        "",
        f"`{output_dir.resolve()}`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    build_report(resolve(args.metrics_dir), resolve(args.output_dir), resolve(args.report_path))


if __name__ == "__main__":
    main()
