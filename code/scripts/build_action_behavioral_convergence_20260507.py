#!/usr/bin/env python3
"""Build action behavioral-convergence diagnostics.

This script treats buy/hold/sell as a behavioral policy output, not as a
portfolio-performance endpoint. It complements the existing quality guardrail
tables by adding action concentration, Simpson/HHI diversity, profile behavior,
high-risk-event concentration, transitions, and cross-dimensional associations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "t4_followup_20260505"
    / "metrics_stage2_plus_t4_four_models_profile_expanded_20260505"
)
DEFAULT_REASONING_DIR = (
    PROJECT_ROOT
    / "docs"
    / "experiment_results"
    / "tables"
    / "reasoning_semantic_diversity_20260507"
)
DEFAULT_HIGH_RISK_MANIFEST = (
    PROJECT_ROOT
    / "docs"
    / "experiment_results"
    / "tables"
    / "high_risk16_t4_comparison_20260506"
    / "manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "docs"
    / "experiment_results"
    / "tables"
    / "action_behavioral_convergence_20260507"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "docs"
    / "experiment_results"
    / "action_behavioral_convergence_20260507.md"
)

ACTION_ORDER = ("buy", "hold", "sell")
TREATMENT_ORDER = ("T1", "T2", "T3", "T4")
INVESTOR_GROUPS = ("all_profiles", "retail", "institutional")
CONTRASTS = (
    ("T2_minus_T1", "T2", "T1"),
    ("T3_minus_T2", "T3", "T2"),
    ("T4_minus_T2", "T4", "T2"),
    ("T4_minus_T3", "T4", "T3"),
    ("T4_minus_T1", "T4", "T1"),
)
ACTION_METRICS = (
    ("action_entropy_bits", "Action entropy bits"),
    ("action_entropy_normalized", "Normalized action entropy"),
    ("action_hhi", "Action HHI concentration"),
    ("action_simpson_diversity", "Simpson action diversity"),
    ("pairwise_same_action_share", "Pairwise same-action share"),
    ("pairwise_action_diversity", "Pairwise action diversity"),
    ("trade_rate", "Trade rate"),
    ("hold_share", "Hold share"),
    ("absolute_net_direction", "Absolute net direction"),
)
CELL_KEY = (
    "event_id",
    "treatment_family",
    "upstream_model_family",
    "model_family",
    "representation_seed",
    "decision_seed",
    "investor_group",
)
TRANSITION_SPECS = (
    ("T1_to_T2", "T1", "T2", ["event_id", "model_family", "decision_seed"]),
    ("T1_to_T3", "T1", "T3", ["event_id", "model_family", "decision_seed"]),
    ("T1_to_T4", "T1", "T4", ["event_id", "model_family", "decision_seed"]),
    (
        "T2_to_T3",
        "T2",
        "T3",
        ["event_id", "model_family", "upstream_model_family", "representation_seed", "decision_seed"],
    ),
    (
        "T2_to_T4",
        "T2",
        "T4",
        ["event_id", "model_family", "upstream_model_family", "representation_seed", "decision_seed"],
    ),
    (
        "T3_to_T4",
        "T3",
        "T4",
        ["event_id", "model_family", "upstream_model_family", "representation_seed", "decision_seed"],
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", default=str(DEFAULT_METRICS_DIR))
    parser.add_argument("--reasoning-dir", default=str(DEFAULT_REASONING_DIR))
    parser.add_argument("--high-risk-manifest", default=str(DEFAULT_HIGH_RISK_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    return parser.parse_args()


def resolve_path(path_text: str | Path) -> Path:
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


def fmt_signed(value: Any, digits: int = 5) -> str:
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
    body = [
        "| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def load_high_risk_events(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(event_id) for event_id in payload.get("event_ids", [])}


def action_code(action: Any) -> int:
    value = str(action).strip().lower()
    if value == "buy":
        return 1
    if value == "sell":
        return -1
    return 0


def shannon_entropy_bits(action_counts: dict[str, int]) -> float:
    total = sum(action_counts.values())
    if total <= 0:
        return math.nan
    entropy = 0.0
    for count in action_counts.values():
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def hhi(action_counts: dict[str, int]) -> float:
    total = sum(action_counts.values())
    if total <= 0:
        return math.nan
    return sum((count / total) ** 2 for count in action_counts.values())


def finite_pairwise_same_share(action_counts: dict[str, int]) -> float:
    total = sum(action_counts.values())
    denominator = total * (total - 1) / 2
    if denominator <= 0:
        return math.nan
    numerator = sum(count * (count - 1) / 2 for count in action_counts.values())
    return numerator / denominator


def bh_fdr_adjust(p_values: list[float]) -> list[float]:
    indexed = [
        (index, value)
        for index, value in enumerate(p_values)
        if value is not None and math.isfinite(float(value))
    ]
    adjusted = [math.nan] * len(p_values)
    if not indexed:
        return adjusted
    ordered = sorted(indexed, key=lambda item: item[1])
    m = len(ordered)
    running = 1.0
    for rank_from_end, (index, p_value) in enumerate(reversed(ordered), start=1):
        rank = m - rank_from_end + 1
        running = min(running, float(p_value) * m / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def bootstrap_event_delta(values: pd.Series, reps: int, seed: int) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return {
            "mean_delta": math.nan,
            "median_delta": math.nan,
            "positive_share": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "p_two_sided": math.nan,
            "n_events": 0,
        }
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(reps, len(clean)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    p_two = 2 * min(float((samples <= 0).mean()), float((samples >= 0).mean()))
    return {
        "mean_delta": float(clean.mean()),
        "median_delta": float(np.median(clean)),
        "positive_share": float((clean > 0).mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_two_sided": min(1.0, p_two),
        "n_events": int(len(clean)),
    }


def sample_label(event_id: Any, high_risk_events: set[str]) -> str:
    return "high_risk16" if str(event_id) in high_risk_events else "non_high_risk"


def rows_for_sample(rows: pd.DataFrame, sample: str) -> pd.DataFrame:
    if sample == "full":
        return rows
    return rows[rows["sample"] == sample]


def build_action_cell_metrics(decisions: pd.DataFrame, high_risk_events: set[str]) -> pd.DataFrame:
    working = decisions.copy()
    working["_action"] = working["action"].astype(str).str.lower()
    working["_action_code"] = working["_action"].apply(action_code)

    rows: list[dict[str, Any]] = []
    for key, group in working.groupby(list(CELL_KEY[:-1]), dropna=False):
        key_payload = dict(zip(CELL_KEY[:-1], key, strict=True))
        for investor_group in INVESTOR_GROUPS:
            if investor_group == "all_profiles":
                subset = group[group["profile_group"].isin(["retail", "institutional"])]
            else:
                subset = group[group["profile_group"].eq(investor_group)]
            if subset.empty:
                continue
            counts = {
                action: int(subset["_action"].eq(action).sum())
                for action in ACTION_ORDER
            }
            total = int(len(subset))
            entropy = shannon_entropy_bits(counts)
            concentration = hhi(counts)
            same_share = finite_pairwise_same_share(counts)
            row = {
                **key_payload,
                "investor_group": investor_group,
                "sample": sample_label(key_payload["event_id"], high_risk_events),
                "profile_count": total,
                "buy_count": counts["buy"],
                "hold_count": counts["hold"],
                "sell_count": counts["sell"],
                "buy_share": counts["buy"] / total,
                "hold_share": counts["hold"] / total,
                "sell_share": counts["sell"] / total,
                "trade_rate": (counts["buy"] + counts["sell"]) / total,
                "action_entropy_bits": entropy,
                "action_entropy_normalized": entropy / math.log2(3),
                "action_hhi": concentration,
                "action_simpson_diversity": 1.0 - concentration,
                "pairwise_same_action_share": same_share,
                "pairwise_action_diversity": 1.0 - same_share if math.isfinite(same_share) else math.nan,
                "absolute_net_direction": abs(float(subset["_action_code"].mean())),
                "action_direction_diversity": 1.0 - abs(float(subset["_action_code"].mean())),
                "mean_action_strength": float(pd.to_numeric(subset["action_strength"], errors="coerce").mean()),
                "mean_confidence": float(pd.to_numeric(subset["confidence"], errors="coerce").mean()),
                "directional_accuracy": float(pd.to_numeric(subset["directional_accuracy"], errors="coerce").mean()),
                "strict_action_accuracy": float(pd.to_numeric(subset["action_accuracy"], errors="coerce").mean()),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def build_treatment_means(action_cells: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric for metric, _label in ACTION_METRICS] + [
        "buy_share",
        "sell_share",
        "mean_action_strength",
        "mean_confidence",
        "directional_accuracy",
        "strict_action_accuracy",
    ]
    rows: list[dict[str, Any]] = []
    for sample in ("full", "high_risk16", "non_high_risk"):
        sample_rows = rows_for_sample(action_cells, sample)
        for (treatment, investor_group), group in sample_rows.groupby(
            ["treatment_family", "investor_group"], dropna=False
        ):
            row: dict[str, Any] = {
                "sample": sample,
                "treatment": treatment,
                "investor_group": investor_group,
                "n_cells": int(len(group)),
                "n_events": int(group["event_id"].nunique()),
            }
            for metric in metrics:
                row[metric] = float(pd.to_numeric(group[metric], errors="coerce").mean())
            rows.append(row)
    output = pd.DataFrame(rows)
    output["treatment_order"] = output["treatment"].apply(
        lambda value: TREATMENT_ORDER.index(value) if value in TREATMENT_ORDER else 999
    )
    output["group_order"] = output["investor_group"].apply(
        lambda value: INVESTOR_GROUPS.index(value) if value in INVESTOR_GROUPS else 999
    )
    return output.sort_values(["sample", "group_order", "treatment_order"]).drop(
        columns=["group_order", "treatment_order"]
    )


def build_action_contrasts(action_cells: pd.DataFrame, reps: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [metric for metric, _label in ACTION_METRICS]
    seed = 20260507
    for sample in ("full", "high_risk16", "non_high_risk"):
        sample_rows = rows_for_sample(action_cells, sample)
        for investor_group in INVESTOR_GROUPS:
            group_rows = sample_rows[sample_rows["investor_group"].eq(investor_group)]
            if group_rows.empty:
                continue
            event_treatment = (
                group_rows.groupby(["event_id", "treatment_family"], dropna=False)[metrics]
                .mean()
                .reset_index()
            )
            for contrast, comparison, baseline in CONTRASTS:
                wide = event_treatment.pivot(index="event_id", columns="treatment_family", values=metrics)
                for metric in metrics:
                    if comparison not in wide[metric].columns or baseline not in wide[metric].columns:
                        continue
                    delta = wide[metric][comparison] - wide[metric][baseline]
                    result = bootstrap_event_delta(delta, reps, seed)
                    seed += 1
                    rows.append(
                        {
                            "sample": sample,
                            "investor_group": investor_group,
                            "contrast": contrast,
                            "comparison": comparison,
                            "baseline": baseline,
                            "metric": metric,
                            **result,
                        }
                    )
    output = pd.DataFrame(rows)
    adjusted_parts: list[pd.DataFrame] = []
    for (_sample, _group), part in output.groupby(["sample", "investor_group"], dropna=False):
        part = part.copy()
        part["p_bh_fdr_action_family"] = bh_fdr_adjust(part["p_two_sided"].tolist())
        adjusted_parts.append(part)
    return pd.concat(adjusted_parts, ignore_index=True)


def build_profile_behavior(decisions: pd.DataFrame, high_risk_events: set[str]) -> pd.DataFrame:
    working = decisions.copy()
    working["_action"] = working["action"].astype(str).str.lower()
    working["_sample"] = working["event_id"].apply(lambda value: sample_label(value, high_risk_events))
    working["action_strength_numeric"] = pd.to_numeric(working["action_strength"], errors="coerce")
    for action in ACTION_ORDER:
        working[f"{action}_share"] = working["_action"].eq(action).astype(float)
    rows: list[pd.DataFrame] = []
    for sample in ("full", "high_risk16", "non_high_risk"):
        subset = working if sample == "full" else working[working["_sample"].eq(sample)]
        grouped = (
            subset.groupby(["profile_canonical", "profile_group", "treatment_family"], dropna=False)
            .agg(
                rows=("row_number", "count"),
                events=("event_id", "nunique"),
                buy_share=("buy_share", "mean"),
                hold_share=("hold_share", "mean"),
                sell_share=("sell_share", "mean"),
                mean_expected_return=("expected_return_5d", "mean"),
                mean_confidence=("confidence", "mean"),
                mean_action_strength=("action_strength_numeric", "mean"),
                directional_accuracy=("directional_accuracy", "mean"),
                strict_action_accuracy=("action_accuracy", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "sample", sample)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def build_profile_contrasts(profile_behavior: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "buy_share",
        "hold_share",
        "sell_share",
        "mean_expected_return",
        "mean_confidence",
        "mean_action_strength",
        "directional_accuracy",
        "strict_action_accuracy",
    )
    rows: list[dict[str, Any]] = []
    for (sample, profile), group in profile_behavior.groupby(["sample", "profile_canonical"], dropna=False):
        indexed = group.set_index("treatment_family")
        for contrast, comparison, baseline in CONTRASTS:
            if comparison not in indexed.index or baseline not in indexed.index:
                continue
            row: dict[str, Any] = {
                "sample": sample,
                "profile": profile,
                "profile_group": indexed.loc[comparison, "profile_group"],
                "contrast": contrast,
            }
            for metric in metrics:
                row[f"{metric}_delta"] = float(indexed.loc[comparison, metric] - indexed.loc[baseline, metric])
            rows.append(row)
    return pd.DataFrame(rows)


def transition_pairs(rows: pd.DataFrame, baseline: str, comparison: str, join_cols: list[str]) -> pd.DataFrame:
    left = rows[rows["treatment_family"].eq(baseline)][
        join_cols + ["profile_canonical", "_action", "_sample"]
    ].rename(columns={"_action": "from_action", "_sample": "from_sample"})
    right = rows[rows["treatment_family"].eq(comparison)][
        join_cols + ["profile_canonical", "_action", "_sample"]
    ].rename(columns={"_action": "to_action", "_sample": "to_sample"})
    merged = right.merge(left, on=join_cols + ["profile_canonical"], how="inner")
    merged["sample"] = merged["to_sample"]
    return merged


def build_transition_summary(decisions: pd.DataFrame, high_risk_events: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = decisions.copy()
    working["_action"] = working["action"].astype(str).str.lower()
    working["_sample"] = working["event_id"].apply(lambda value: sample_label(value, high_risk_events))
    pair_frames: list[pd.DataFrame] = []
    for contrast, baseline, comparison, join_cols in TRANSITION_SPECS:
        pairs = transition_pairs(working, baseline, comparison, join_cols)
        pairs.insert(0, "contrast", contrast)
        pair_frames.append(pairs)
    all_pairs = pd.concat(pair_frames, ignore_index=True)

    rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    for sample in ("full", "high_risk16", "non_high_risk"):
        sample_pairs = all_pairs if sample == "full" else all_pairs[all_pairs["sample"].eq(sample)]
        for contrast, group in sample_pairs.groupby("contrast", dropna=False):
            total = len(group)
            same = int(group["from_action"].eq(group["to_action"]).sum())
            rows.append(
                {
                    "sample": sample,
                    "contrast": contrast,
                    "n_pairs": total,
                    "same_action_share": same / total if total else math.nan,
                    "changed_action_share": 1 - same / total if total else math.nan,
                    "buy_to_hold_share": transition_share(group, "buy", "hold"),
                    "sell_to_hold_share": transition_share(group, "sell", "hold"),
                    "hold_to_trade_share": float(
                        (
                            group["from_action"].eq("hold")
                            & group["to_action"].isin(["buy", "sell"])
                        ).mean()
                    )
                    if total
                    else math.nan,
                    "trade_flip_share": float(
                        (
                            (group["from_action"].eq("buy") & group["to_action"].eq("sell"))
                            | (group["from_action"].eq("sell") & group["to_action"].eq("buy"))
                        ).mean()
                    )
                    if total
                    else math.nan,
                }
            )
            for profile, profile_group in group.groupby("profile_canonical", dropna=False):
                profile_total = len(profile_group)
                profile_same = int(profile_group["from_action"].eq(profile_group["to_action"]).sum())
                profile_rows.append(
                    {
                        "sample": sample,
                        "contrast": contrast,
                        "profile": profile,
                        "n_pairs": profile_total,
                        "same_action_share": profile_same / profile_total if profile_total else math.nan,
                        "changed_action_share": 1 - profile_same / profile_total if profile_total else math.nan,
                        "buy_to_hold_share": transition_share(profile_group, "buy", "hold"),
                        "sell_to_hold_share": transition_share(profile_group, "sell", "hold"),
                        "hold_to_trade_share": float(
                            (
                                profile_group["from_action"].eq("hold")
                                & profile_group["to_action"].isin(["buy", "sell"])
                            ).mean()
                        )
                        if profile_total
                        else math.nan,
                        "trade_flip_share": float(
                            (
                                (
                                    profile_group["from_action"].eq("buy")
                                    & profile_group["to_action"].eq("sell")
                                )
                                | (
                                    profile_group["from_action"].eq("sell")
                                    & profile_group["to_action"].eq("buy")
                                )
                            ).mean()
                        )
                        if profile_total
                        else math.nan,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(profile_rows)


def transition_share(group: pd.DataFrame, from_action: str, to_action: str) -> float:
    if len(group) == 0:
        return math.nan
    return float((group["from_action"].eq(from_action) & group["to_action"].eq(to_action)).mean())


def build_cross_dimensional_correlations(
    action_cells: pd.DataFrame,
    cell_metrics: pd.DataFrame,
    reasoning_cells: pd.DataFrame | None,
) -> pd.DataFrame:
    merge_cols = list(CELL_KEY)
    merged = action_cells.merge(
        cell_metrics[
            merge_cols
            + [
                "mean_pairwise_abs_diff_expected_return_5d",
                "source_id_diversity_one_minus_overlap",
                "evidence_category_diversity_one_minus_overlap",
                "mean_pairwise_sentence_embedding_distance",
            ]
        ],
        on=merge_cols,
        how="left",
    )
    if reasoning_cells is not None and not reasoning_cells.empty:
        merged = merged.merge(
            reasoning_cells[
                merge_cols
                + [
                    "expressed_reason_semantic_distance",
                    "uncertainty_frame_semantic_distance",
                    "composite_reasoning_semantic_distance",
                ]
            ],
            on=merge_cols,
            how="left",
        )

    action_vars = ("action_hhi", "action_simpson_diversity", "pairwise_same_action_share", "hold_share")
    other_vars = (
        "source_id_diversity_one_minus_overlap",
        "evidence_category_diversity_one_minus_overlap",
        "mean_pairwise_sentence_embedding_distance",
        "expressed_reason_semantic_distance",
        "uncertainty_frame_semantic_distance",
        "composite_reasoning_semantic_distance",
        "mean_pairwise_abs_diff_expected_return_5d",
    )
    rows: list[dict[str, Any]] = []
    for treatment in ("all", *TREATMENT_ORDER):
        subset = merged if treatment == "all" else merged[merged["treatment_family"].eq(treatment)]
        subset = subset[subset["investor_group"].eq("all_profiles")]
        for action_var in action_vars:
            for other_var in other_vars:
                if other_var not in subset.columns:
                    continue
                clean = subset[[action_var, other_var]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(clean) < 3:
                    corr = math.nan
                else:
                    corr = float(clean[action_var].corr(clean[other_var], method="spearman"))
                rows.append(
                    {
                        "treatment": treatment,
                        "action_metric": action_var,
                        "paired_metric": other_var,
                        "spearman_r": corr,
                        "n_cells": int(len(clean)),
                    }
                )
    return pd.DataFrame(rows)


def build_rq_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rq_or_hypothesis": "RQ3",
                "question": "Does convergence appear in beliefs, actions, rationales, evidence use, or only some layers?",
                "added_action_output": "action_treatment_means.csv; action_primary_contrasts_fdr.csv",
                "interpretation": "Action is treated as downstream behavioral policy-output convergence, parallel to belief and rationale.",
            },
            {
                "rq_or_hypothesis": "RQ4/H6",
                "question": "Is convergence useful, unsupported, or harmful?",
                "added_action_output": "action_primary_contrasts_fdr.csv plus existing quality guardrails",
                "interpretation": "Behavioral concentration is interpreted with directional/trade/hold guardrails rather than as portfolio welfare.",
            },
            {
                "rq_or_hypothesis": "RQ5/H5",
                "question": "Do effects vary by investor group or profile?",
                "added_action_output": "action_profile_behavior.csv; action_profile_contrasts.csv",
                "interpretation": "Profile-specific policy shifts test whether shared interfaces compress role-specific behavior.",
            },
            {
                "rq_or_hypothesis": "H4",
                "question": "Do information environments moderate convergence?",
                "added_action_output": "high_risk16 rows in all action tables",
                "interpretation": "The pre-existing high-risk16 diagnostic checks whether action concentration is stronger in actionable-risk events.",
            },
            {
                "rq_or_hypothesis": "Cross-layer transmission",
                "question": "Do evidence/reasoning compression and action concentration move together?",
                "added_action_output": "action_cross_dimensional_correlations.csv",
                "interpretation": "Associational only; useful for mechanism discussion, not causal mediation.",
            },
        ]
    )


def selected_contrast_rows(contrasts: pd.DataFrame) -> list[dict[str, str]]:
    wanted = contrasts[
        (contrasts["sample"].eq("full"))
        & (contrasts["investor_group"].eq("all_profiles"))
        & (
            contrasts["metric"].isin(
                [
                    "action_entropy_bits",
                    "action_simpson_diversity",
                    "pairwise_same_action_share",
                    "hold_share",
                ]
            )
        )
        & (contrasts["contrast"].isin(["T3_minus_T2", "T4_minus_T2", "T4_minus_T3"]))
    ].copy()
    wanted["order"] = wanted["contrast"].apply(
        lambda value: ["T3_minus_T2", "T4_minus_T2", "T4_minus_T3"].index(value)
    )
    wanted = wanted.sort_values(["order", "metric"])
    return [
        {
            "Contrast": row["contrast"],
            "Metric": row["metric"],
            "Delta": fmt_signed(row["mean_delta"]),
            "95% CI": f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]",
            "p": fmt(row["p_two_sided"], 4),
            "BH-FDR": fmt(row["p_bh_fdr_action_family"], 4),
        }
        for row in wanted.to_dict("records")
    ]


def selected_high_risk_rows(means: pd.DataFrame) -> list[dict[str, str]]:
    wanted = means[(means["sample"].eq("high_risk16")) & (means["investor_group"].eq("all_profiles"))]
    return [
        {
            "Treatment": row["treatment"],
            "Entropy": fmt(row["action_entropy_bits"]),
            "Simpson": fmt(row["action_simpson_diversity"]),
            "Same action": fmt(row["pairwise_same_action_share"]),
            "Hold": fmt(row["hold_share"]),
            "Trade": fmt(row["trade_rate"]),
        }
        for row in wanted.to_dict("records")
    ]


def selected_transition_rows(transitions: pd.DataFrame) -> list[dict[str, str]]:
    wanted = transitions[
        (transitions["sample"].eq("full"))
        & (transitions["contrast"].isin(["T1_to_T2", "T2_to_T3", "T2_to_T4"]))
    ].copy()
    wanted["order"] = wanted["contrast"].apply(
        lambda value: ["T1_to_T2", "T2_to_T3", "T2_to_T4"].index(value)
    )
    wanted = wanted.sort_values("order")
    return [
        {
            "Transition": row["contrast"],
            "Same": fmt(row["same_action_share"]),
            "Changed": fmt(row["changed_action_share"]),
            "Buy->Hold": fmt(row["buy_to_hold_share"]),
            "Sell->Hold": fmt(row["sell_to_hold_share"]),
            "Hold->Trade": fmt(row["hold_to_trade_share"]),
            "Trade flip": fmt(row["trade_flip_share"]),
        }
        for row in wanted.to_dict("records")
    ]


def write_report(
    path: Path,
    means: pd.DataFrame,
    contrasts: pd.DataFrame,
    transitions: pd.DataFrame,
    output_dir: Path,
) -> None:
    all_full = means[(means["sample"].eq("full")) & (means["investor_group"].eq("all_profiles"))]
    mean_rows = [
        {
            "Treatment": row["treatment"],
            "Entropy": fmt(row["action_entropy_bits"]),
            "Simpson": fmt(row["action_simpson_diversity"]),
            "Same action": fmt(row["pairwise_same_action_share"]),
            "HHI": fmt(row["action_hhi"]),
            "Hold": fmt(row["hold_share"]),
            "Trade": fmt(row["trade_rate"]),
        }
        for row in all_full.to_dict("records")
    ]
    body = f"""# Action Behavioral-Convergence Diagnostic - 2026-05-07

## Purpose

This addendum promotes action from a narrow accuracy proxy to a downstream
behavioral outcome. The estimand is policy-output concentration among synthetic
decision-makers, not realized portfolio welfare.

## Formulas

```text
p_k = share of action k, where k in {{buy, hold, sell}}
H_action = -sum_k p_k log2(p_k)
HHI_action = sum_k p_k^2
Simpson_action_diversity = 1 - HHI_action
pairwise_same_action_share = sum_k choose(n_k, 2) / choose(n, 2)
pairwise_action_diversity = 1 - pairwise_same_action_share
```

`HHI_action` is concentration. `Simpson_action_diversity` is the probability
that two independent draws from the action distribution differ. The finite
pairwise version reports the same idea without replacement among the observed
profile panel.

## Full-Sample Action Means

{md_table(mean_rows, [
    ("Treatment", "Treatment"),
    ("Entropy", "Entropy"),
    ("Simpson", "Simpson"),
    ("Same action", "Same action"),
    ("HHI", "HHI"),
    ("Hold", "Hold"),
    ("Trade", "Trade"),
])}

## Primary Action Contrasts With BH-FDR

{md_table(selected_contrast_rows(contrasts), [
    ("Contrast", "Contrast"),
    ("Metric", "Metric"),
    ("Delta", "Delta"),
    ("95% CI", "95% CI"),
    ("p", "p"),
    ("BH-FDR", "BH-FDR"),
])}

## High-Risk16 Action Concentration

{md_table(selected_high_risk_rows(means), [
    ("Treatment", "Treatment"),
    ("Entropy", "Entropy"),
    ("Simpson", "Simpson"),
    ("Same action", "Same action"),
    ("Hold", "Hold"),
    ("Trade", "Trade"),
])}

## Transition Summary

{md_table(selected_transition_rows(transitions), [
    ("Transition", "Transition"),
    ("Same", "Same"),
    ("Changed", "Changed"),
    ("Buy->Hold", "Buy->Hold"),
    ("Sell->Hold", "Sell->Hold"),
    ("Hold->Trade", "Hold->Trade"),
    ("Trade flip", "Trade flip"),
])}

## RQ/Hypothesis Contribution

Action metrics now answer RQ3 and H5 directly: the experiment can distinguish
evidence/reasoning convergence from behavioral policy-output convergence and
can test whether profile-specific behavior survives shared evidence interfaces.
They also deepen RQ4/H6 because behavioral concentration is interpreted against
quality guardrails rather than assumed to be useful or harmful.

## Interpretation Boundary

Buy/hold/sell remains a coarse policy output. The action section should claim
behavioral convergence, profile shifts, and concentration patterns. It should
not claim portfolio welfare unless supported by separate quality or human
validation evidence.

## Theoretical Anchors

- Shannon entropy measures uncertainty in a categorical distribution
  ([Shannon, 1948](https://doi.org/10.1002/j.1538-7305.1948.tb00917.x)).
- Simpson diversity and HHI-style concentration measure how mass is distributed
  across categories ([Simpson, 1949](https://doi.org/10.1038/163688a0);
  [DOJ HHI convention](https://www.justice.gov/atr/herfindahl-hirschman-index)).
- BH-FDR controls the expected false-discovery rate across the action metric
  family more appropriately than Bonferroni for correlated diagnostics
  ([Benjamini and Hochberg, 1995](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)).
- Hierarchical models remain the right paper-level robustness frame for nested
  event/profile/model observations, but the table here uses event-paired
  bootstrap plus FDR as the reproducible first pass.

Artifacts:

- `{output_dir / "action_cell_metrics.csv"}`
- `{output_dir / "action_treatment_means.csv"}`
- `{output_dir / "action_primary_contrasts_fdr.csv"}`
- `{output_dir / "action_profile_behavior.csv"}`
- `{output_dir / "action_profile_contrasts.csv"}`
- `{output_dir / "action_transition_summary.csv"}`
- `{output_dir / "action_transition_by_profile.csv"}`
- `{output_dir / "action_cross_dimensional_correlations.csv"}`
- `{output_dir / "rq_hypothesis_action_mapping.csv"}`
"""
    path.write_text(body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics_dir = resolve_path(args.metrics_dir)
    reasoning_dir = resolve_path(args.reasoning_dir)
    output_dir = resolve_path(args.output_dir)
    report_path = resolve_path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)

    high_risk_events = load_high_risk_events(resolve_path(args.high_risk_manifest))
    decisions = pd.read_csv(metrics_dir / "decision_rows.csv")
    cell_metrics = pd.read_csv(metrics_dir / "cell_metrics.csv")
    reasoning_path = reasoning_dir / "reasoning_cell_metrics.csv"
    reasoning_cells = pd.read_csv(reasoning_path) if reasoning_path.exists() else None

    action_cells = build_action_cell_metrics(decisions, high_risk_events)
    treatment_means = build_treatment_means(action_cells)
    contrasts = build_action_contrasts(action_cells, args.bootstrap_reps)
    profile_behavior = build_profile_behavior(decisions, high_risk_events)
    profile_contrasts = build_profile_contrasts(profile_behavior)
    transition_summary, transition_by_profile = build_transition_summary(decisions, high_risk_events)
    cross_dimensional = build_cross_dimensional_correlations(action_cells, cell_metrics, reasoning_cells)
    rq_mapping = build_rq_mapping()

    action_cells.to_csv(output_dir / "action_cell_metrics.csv", index=False)
    treatment_means.to_csv(output_dir / "action_treatment_means.csv", index=False)
    contrasts.to_csv(output_dir / "action_primary_contrasts_fdr.csv", index=False)
    profile_behavior.to_csv(output_dir / "action_profile_behavior.csv", index=False)
    profile_contrasts.to_csv(output_dir / "action_profile_contrasts.csv", index=False)
    transition_summary.to_csv(output_dir / "action_transition_summary.csv", index=False)
    transition_by_profile.to_csv(output_dir / "action_transition_by_profile.csv", index=False)
    cross_dimensional.to_csv(output_dir / "action_cross_dimensional_correlations.csv", index=False)
    rq_mapping.to_csv(output_dir / "rq_hypothesis_action_mapping.csv", index=False)
    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "decision_rows": str(metrics_dir / "decision_rows.csv"),
                "cell_metrics": str(metrics_dir / "cell_metrics.csv"),
                "reasoning_cells": str(reasoning_path) if reasoning_path.exists() else None,
                "high_risk_event_count": len(high_risk_events),
                "bootstrap_reps": args.bootstrap_reps,
                "action_cell_rows": len(action_cells),
                "contrast_rows": len(contrasts),
                "profile_behavior_rows": len(profile_behavior),
                "transition_rows": len(transition_summary),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(report_path, treatment_means, contrasts, transition_summary, output_dir)

    print(f"action_cell_rows={len(action_cells)}")
    print(f"contrast_rows={len(contrasts)}")
    print(f"output_dir={output_dir}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
