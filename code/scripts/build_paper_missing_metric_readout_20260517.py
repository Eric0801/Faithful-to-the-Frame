#!/usr/bin/env python3
"""Build paper-facing readouts for missing E2A metric details.

This script uses the same downstream metric bundle used by the current paper
draft:

data/clean_strict_predata_2026_main/t4_followup_20260505/
  metrics_stage2_plus_t4_plus_b0_four_models_20260510/

It intentionally reports source uptake in both raw-row and event-level forms,
because source uptake is a per-decision quantity while most diversity metrics
in the paper are event-level cell aggregates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRIC_DIR = (
    ROOT
    / "data/clean_strict_predata_2026_main/t4_followup_20260505/"
    "metrics_stage2_plus_t4_plus_b0_four_models_20260510"
)
QUALITY_DIR = ROOT / "docs/experiment_results/tables/quality_guardrail_with_t4_20260506"
OUT_DIR = ROOT / "docs/experiment_results/tables/paper_missing_metrics_20260517"
REPORT_PATH = ROOT / "docs/paper_writing/notes/missing_metric_readout_20260517.md"
BOOTSTRAP_REPS = 10_000
SEED = 20260517


TREATMENT_ORDER = ["T1", "B0", "T2", "T3", "T4"]


def parse_json_list(value: object) -> list[object]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    return json.loads(value)


def bootstrap_ci(values: Iterable[float], *, reps: int = BOOTSTRAP_REPS) -> tuple[float, float]:
    arr = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if len(arr) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(SEED)
    draws = rng.choice(arr, size=(reps, len(arr)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def bootstrap_p_two_sided(values: Iterable[float], *, reps: int = BOOTSTRAP_REPS) -> float:
    arr = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if len(arr) == 0:
        return np.nan
    rng = np.random.default_rng(SEED)
    draws = rng.choice(arr, size=(reps, len(arr)), replace=True).mean(axis=1)
    return float(2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))


def treatment_mean_ci(event_metric: pd.DataFrame, metric: str, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for treatment in TREATMENT_ORDER:
        sub = event_metric[event_metric["treatment_family"] == treatment]
        if sub.empty:
            continue
        ci_low, ci_high = bootstrap_ci(sub[metric])
        rows.append(
            {
                "metric": label,
                "treatment_family": treatment,
                "n_events": int(sub["event_id"].nunique()),
                "event_level_mean": sub[metric].mean(),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def treatment_contrast_ci(
    event_metric: pd.DataFrame,
    metric: str,
    label: str,
    contrasts: list[tuple[str, str]],
) -> pd.DataFrame:
    pivot = event_metric.pivot(index="event_id", columns="treatment_family", values=metric)
    rows: list[dict[str, object]] = []
    for comparison, baseline in contrasts:
        if comparison not in pivot or baseline not in pivot:
            continue
        delta = (pivot[comparison] - pivot[baseline]).dropna()
        ci_low, ci_high = bootstrap_ci(delta)
        rows.append(
            {
                "metric": label,
                "contrast": f"{comparison}_minus_{baseline}",
                "comparison": comparison,
                "baseline": baseline,
                "n_events": int(delta.shape[0]),
                "mean_delta": delta.mean(),
                "median_delta": delta.median(),
                "positive_share": float((delta > 0).mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_two_sided": bootstrap_p_two_sided(delta),
            }
        )
    return pd.DataFrame(rows)


def source_uptake(decisions: pd.DataFrame) -> pd.DataFrame:
    df = decisions.copy()
    df["evidence_count"] = df["evidence_used"].map(lambda value: len(parse_json_list(value)))
    rows: list[dict[str, object]] = []
    for treatment in TREATMENT_ORDER:
        sub = df[df["treatment_family"] == treatment]
        if sub.empty:
            continue
        event_mean = (
            sub.groupby(["event_id", "treatment_family"], as_index=False)["evidence_count"]
            .mean()
            .rename(columns={"evidence_count": "event_mean_evidence_count"})
        )
        ci_low, ci_high = bootstrap_ci(event_mean["event_mean_evidence_count"])
        rows.append(
            {
                "treatment_family": treatment,
                "n_decisions": int(sub.shape[0]),
                "n_events": int(sub["event_id"].nunique()),
                "raw_row_mean_cited_sources": sub["evidence_count"].mean(),
                "event_level_mean_cited_sources": event_mean[
                    "event_mean_evidence_count"
                ].mean(),
                "event_level_ci_low": ci_low,
                "event_level_ci_high": ci_high,
                "raw_row_median_cited_sources": sub["evidence_count"].median(),
            }
        )
    return pd.DataFrame(rows)


def positive_sign_event_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    df = decisions.copy()
    df["positive_expected_return"] = (df["expected_return_5d"] > 0).astype(float)
    return (
        df.groupby(["event_id", "treatment_family"], as_index=False)
        .agg(
            positive_sign_share=("positive_expected_return", "mean"),
            mean_expected_return_5d=("expected_return_5d", "mean"),
            n_decisions=("expected_return_5d", "size"),
        )
    )


def profile_separability(decisions: pd.DataFrame) -> pd.DataFrame:
    df = decisions.copy()
    df["evidence_count"] = df["evidence_used"].map(lambda value: len(parse_json_list(value)))
    df["category_count"] = df["evidence_categories"].map(lambda value: len(parse_json_list(value)))
    for action in ["buy", "hold", "sell"]:
        df[f"{action}_share"] = (df["action"] == action).astype(float)

    profile_event = (
        df.groupby(["event_id", "treatment_family", "profile_canonical"], as_index=False)
        .agg(
            expected_return_5d=("expected_return_5d", "mean"),
            evidence_count=("evidence_count", "mean"),
            category_count=("category_count", "mean"),
            buy_share=("buy_share", "mean"),
            hold_share=("hold_share", "mean"),
            sell_share=("sell_share", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
            action_accuracy=("action_accuracy", "mean"),
        )
    )
    contrast_specs = [
        ("long_term_retail", "day_trader", "hold_share"),
        ("investment_advisor", "prop_trading", "hold_share"),
        ("day_trader", "long_term_retail", "buy_share"),
        ("prop_trading", "investment_advisor", "buy_share"),
        ("long_term_retail", "day_trader", "evidence_count"),
        ("investment_advisor", "prop_trading", "evidence_count"),
    ]
    rows: list[dict[str, object]] = []
    for treatment in TREATMENT_ORDER:
        sub = profile_event[profile_event["treatment_family"] == treatment]
        for comparison, baseline, metric in contrast_specs:
            pivot = sub.pivot(index="event_id", columns="profile_canonical", values=metric)
            if comparison not in pivot or baseline not in pivot:
                continue
            delta = (pivot[comparison] - pivot[baseline]).dropna()
            ci_low, ci_high = bootstrap_ci(delta)
            rows.append(
                {
                    "treatment_family": treatment,
                    "contrast": f"{comparison}_minus_{baseline}",
                    "comparison_profile": comparison,
                    "baseline_profile": baseline,
                    "metric": metric,
                    "n_events": int(delta.shape[0]),
                    "mean_delta": delta.mean(),
                    "median_delta": delta.median(),
                    "positive_share": float((delta > 0).mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_two_sided": bootstrap_p_two_sided(delta),
                }
            )
    return pd.DataFrame(rows)


def format_float(value: object, digits: int = 5) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[tuple[str, str]], digits: int = 5) -> str:
    header = "| " + " | ".join(title for _, title in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells: list[str] = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, (float, np.floating)):
                cells.append(format_float(value, digits))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    decisions = pd.read_csv(METRIC_DIR / "decision_rows.csv", low_memory=False)
    cells = pd.read_csv(METRIC_DIR / "cell_metrics.csv")
    all_profile_cells = cells[cells["investor_group"] == "all_profiles"].copy()

    event_metrics = (
        all_profile_cells.groupby(["event_id", "treatment_family"], as_index=False)
        .agg(
            category_diversity=("evidence_category_diversity_one_minus_overlap", "mean"),
            belief_mad=("mean_pairwise_abs_diff_expected_return_5d", "mean"),
        )
    )

    uptake = source_uptake(decisions)
    category_means = treatment_mean_ci(event_metrics, "category_diversity", "category_diversity")
    category_contrasts = treatment_contrast_ci(
        event_metrics,
        "category_diversity",
        "category_diversity",
        [("T2", "T1"), ("T3", "T2"), ("T4", "T2"), ("T4", "T3"), ("T4", "B0"), ("B0", "T1")],
    )
    belief_means = treatment_mean_ci(event_metrics, "belief_mad", "belief_mad")
    belief_contrasts = treatment_contrast_ci(
        event_metrics,
        "belief_mad",
        "belief_mad",
        [("T2", "T1"), ("T3", "T2"), ("T4", "T2"), ("T4", "T3"), ("T4", "B0"), ("B0", "T1")],
    )
    positive_sign_events = positive_sign_event_metrics(decisions)
    positive_sign_means = treatment_mean_ci(
        positive_sign_events,
        "positive_sign_share",
        "positive_sign_agreement",
    )
    positive_sign_contrasts = treatment_contrast_ci(
        positive_sign_events,
        "positive_sign_share",
        "positive_sign_agreement",
        [("B0", "T1"), ("T2", "T1"), ("T3", "T2"), ("T2", "B0"), ("T3", "B0"), ("T4", "B0")],
    )

    quality_contrasts = pd.read_csv(QUALITY_DIR / "quality_paired_contrasts_with_t4.csv")
    strict_t2_t1 = quality_contrasts[
        (quality_contrasts["contrast"] == "T2_minus_T1")
        & (quality_contrasts["metric_key"] == "strict_action_accuracy")
    ].copy()

    hold_thresholds = pd.read_csv(QUALITY_DIR / "hold_threshold_sensitivity_with_t4.csv")
    profile_formal = profile_separability(decisions)

    uptake.to_csv(OUT_DIR / "per_decision_source_uptake_by_regime.csv", index=False)
    category_means.to_csv(OUT_DIR / "category_diversity_by_regime_event_bootstrap.csv", index=False)
    category_contrasts.to_csv(OUT_DIR / "category_diversity_contrasts_event_bootstrap.csv", index=False)
    belief_means.to_csv(OUT_DIR / "belief_mad_by_regime_event_bootstrap.csv", index=False)
    belief_contrasts.to_csv(OUT_DIR / "belief_mad_contrasts_event_bootstrap.csv", index=False)
    positive_sign_means.to_csv(
        OUT_DIR / "positive_sign_agreement_by_regime_event_bootstrap.csv",
        index=False,
    )
    positive_sign_contrasts.to_csv(
        OUT_DIR / "positive_sign_agreement_contrasts_event_bootstrap.csv",
        index=False,
    )
    strict_t2_t1.to_csv(OUT_DIR / "strict_action_accuracy_t2_minus_t1_event_bootstrap.csv", index=False)
    hold_thresholds.to_csv(OUT_DIR / "hold_threshold_sensitivity_with_t4.csv", index=False)
    profile_formal.to_csv(OUT_DIR / "profile_separability_formal_event_bootstrap.csv", index=False)

    report_parts = [
        "# Missing Metric Readout - 2026-05-17",
        "",
        "Source: `metrics_stage2_plus_t4_plus_b0_four_models_20260510`. "
        "Diversity metrics use `investor_group == all_profiles`, matching the "
        "current master table. Source uptake is reported as both raw-row mean "
        "and event-level mean.",
        "",
        "## 1. Per-Decision Source Uptake",
        markdown_table(
            uptake,
            [
                ("treatment_family", "Treatment"),
                ("n_decisions", "N decisions"),
                ("n_events", "N events"),
                ("raw_row_mean_cited_sources", "Raw row mean"),
                ("event_level_mean_cited_sources", "Event mean"),
                ("event_level_ci_low", "CI low"),
                ("event_level_ci_high", "CI high"),
                ("raw_row_median_cited_sources", "Raw row median"),
            ],
            digits=3,
        ),
        "",
        "## 2. Category Diversity by Regime",
        markdown_table(
            category_means,
            [
                ("treatment_family", "Treatment"),
                ("n_events", "N events"),
                ("event_level_mean", "Mean"),
                ("ci_low", "CI low"),
                ("ci_high", "CI high"),
            ],
        ),
        "",
        "## 3. Category Diversity Contrasts",
        markdown_table(
            category_contrasts,
            [
                ("contrast", "Contrast"),
                ("n_events", "N events"),
                ("mean_delta", "Delta"),
                ("ci_low", "CI low"),
                ("ci_high", "CI high"),
                ("p_two_sided", "p"),
            ],
        ),
        "",
        "## 4. T2 - T1 Strict Action Accuracy",
        markdown_table(
            strict_t2_t1,
            [
                ("contrast", "Contrast"),
                ("metric", "Metric"),
                ("mean_difference", "Delta"),
                ("ci_low", "CI low"),
                ("ci_high", "CI high"),
                ("p_two_sided", "p"),
                ("n_events", "N events"),
                ("comparison_mean", "T2 mean"),
                ("baseline_mean", "T1 mean"),
            ],
        ),
        "",
        "## 5. Belief MAD by Regime",
        markdown_table(
            belief_means,
            [
                ("treatment_family", "Treatment"),
                ("n_events", "N events"),
                ("event_level_mean", "Mean"),
                ("ci_low", "CI low"),
                ("ci_high", "CI high"),
            ],
        ),
        "",
        "## 6. Belief MAD Contrasts",
        markdown_table(
            belief_contrasts,
            [
                ("contrast", "Contrast"),
                ("n_events", "N events"),
                ("mean_delta", "Delta"),
                ("ci_low", "CI low"),
                ("ci_high", "CI high"),
                ("p_two_sided", "p"),
            ],
        ),
        "",
        "## 7. Positive Sign Agreement by Regime",
        markdown_table(
            positive_sign_means,
            [
                ("treatment_family", "Treatment"),
                ("n_events", "N events"),
                ("event_level_mean", "Mean"),
                ("ci_low", "CI low"),
                ("ci_high", "CI high"),
            ],
        ),
        "",
        "## 8. Positive Sign Agreement Contrasts",
        markdown_table(
            positive_sign_contrasts,
            [
                ("contrast", "Contrast"),
                ("n_events", "N events"),
                ("mean_delta", "Delta"),
                ("ci_low", "CI low"),
                ("ci_high", "CI high"),
                ("p_two_sided", "p"),
            ],
        ),
        "",
        "## 9. Hold-Neutral Threshold Sensitivity",
        markdown_table(
            hold_thresholds,
            [
                ("threshold", "Threshold"),
                ("treatment", "Treatment"),
                ("hold_neutral_accuracy", "Hold-neutral acc"),
                ("ci_low", "CI low"),
                ("ci_high", "CI high"),
                ("n_events", "N events"),
                ("hold_rate", "Hold rate"),
            ],
        ),
        "",
        "## 10. Profile Separability Formal Gate",
        "Event-cluster paired profile contrasts. Positive deltas mean the comparison "
        "profile has a higher value than the baseline profile. The T1 rows are the "
        "clean Appendix B gate; other treatments are supplemental.",
        "",
        markdown_table(
            profile_formal[profile_formal["treatment_family"] == "T1"],
            [
                ("contrast", "Contrast"),
                ("metric", "Metric"),
                ("n_events", "N events"),
                ("mean_delta", "Delta"),
                ("ci_low", "CI low"),
                ("ci_high", "CI high"),
                ("positive_share", "Positive share"),
                ("p_two_sided", "p"),
            ],
        ),
        "",
        "CSV outputs are in "
        "`docs/experiment_results/tables/paper_missing_metrics_20260517/`.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report_parts), encoding="utf-8")


if __name__ == "__main__":
    main()
