#!/usr/bin/env python3
"""Event-cluster bootstrap for B0 mechanism contrasts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRASTS = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "t4_followup_20260505"
    / "metrics_stage2_plus_t4_plus_b0_four_models_20260510"
    / "treatment_contrasts.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "docs"
    / "experiment_results"
    / "tables"
    / "b0_canonical_baseline_20260510"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "docs"
    / "experiment_results"
    / "b0_event_level_bootstrap_contrasts_20260510.md"
)

RATIONALE_METRIC = "mean_pairwise_sentence_embedding_distance"
TARGET_CONTRASTS = ("B0_minus_T1", "T2_minus_B0", "T3_minus_B0")
DEFAULT_REPS = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrast-csv", default=str(DEFAULT_CONTRASTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--seed", type=int, default=20260510)
    return parser.parse_args()


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fmt(value: Any, digits: int = 5) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def fmt_p(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    if number < 0.0001:
        return "<0.0001"
    return f"{number:.4f}"


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def bootstrap_values(values: np.ndarray, *, reps: int, seed: int) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "event_mean_delta": math.nan,
            "event_median_delta": math.nan,
            "event_positive_share": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "p_two_sided": math.nan,
        }
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
    p_two = 2 * min(float((samples <= 0).mean()), float((samples >= 0).mean()))
    return {
        "event_mean_delta": float(values.mean()),
        "event_median_delta": float(np.median(values)),
        "event_positive_share": float((values > 0).mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_two_sided": min(1.0, p_two),
    }


def summarize_event_bootstrap(
    df: pd.DataFrame,
    *,
    contrast_id: str,
    reps: int,
    seed: int,
    model_family: str | None = None,
) -> dict[str, Any]:
    subset = df[
        df["contrast_id"].eq(contrast_id)
        & df["metric_name"].eq(RATIONALE_METRIC)
        & df["investor_group"].eq("all_profiles")
    ].copy()
    if model_family is not None:
        subset = subset[subset["model_family"].eq(model_family)].copy()

    subset["_delta"] = numeric(subset["delta"])
    event_values = subset.dropna(subset=["_delta"]).groupby("event_id")["_delta"].mean()
    summary = bootstrap_values(event_values.to_numpy(dtype=float), reps=reps, seed=seed)
    return {
        "contrast_id": contrast_id,
        "model_family": model_family or "all_models",
        "metric_name": RATIONALE_METRIC,
        "metric_label": "rationale_distance",
        "n_events": int(event_values.shape[0]),
        "n_rows": int(subset["_delta"].notna().sum()),
        "row_mean_delta": float(subset["_delta"].mean()) if subset["_delta"].notna().any() else math.nan,
        **summary,
        "ci_excludes_zero": bool(summary["ci_low"] > 0 or summary["ci_high"] < 0)
        if math.isfinite(summary["ci_low"]) and math.isfinite(summary["ci_high"])
        else False,
    }


def build_outputs(contrast_csv: Path, output_dir: Path, report_path: Path, reps: int, seed: int) -> None:
    contrasts = pd.read_csv(contrast_csv, low_memory=False)

    pooled_rows = []
    for index, contrast_id in enumerate(TARGET_CONTRASTS):
        pooled_rows.append(
            summarize_event_bootstrap(
                contrasts,
                contrast_id=contrast_id,
                reps=reps,
                seed=seed + index,
            )
        )

    model_rows = []
    models = sorted(contrasts["model_family"].dropna().astype(str).unique())
    for contrast_index, contrast_id in enumerate(TARGET_CONTRASTS):
        for model_index, model in enumerate(models):
            model_rows.append(
                summarize_event_bootstrap(
                    contrasts,
                    contrast_id=contrast_id,
                    reps=reps,
                    seed=seed + 100 + contrast_index * 10 + model_index,
                    model_family=model,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    pooled_df = pd.DataFrame(pooled_rows)
    model_df = pd.DataFrame(model_rows)
    pooled_path = output_dir / "b0_event_level_bootstrap_rationale_20260510.csv"
    model_path = output_dir / "b0_event_level_bootstrap_rationale_by_model_20260510.csv"
    pooled_df.to_csv(pooled_path, index=False, lineterminator="\n")
    model_df.to_csv(model_path, index=False, lineterminator="\n")

    display_rows = []
    for row in pooled_rows:
        display_rows.append(
            {
                "Contrast": row["contrast_id"],
                "Rows": row["n_rows"],
                "Events": row["n_events"],
                "Event mean": fmt(row["event_mean_delta"]),
                "Median": fmt(row["event_median_delta"]),
                "Positive share": fmt(row["event_positive_share"], 3),
                "95% CI": f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]",
                "p": fmt_p(row["p_two_sided"]),
                "Excludes 0": "yes" if row["ci_excludes_zero"] else "no",
            }
        )

    by_model_rows = []
    for row in model_rows:
        by_model_rows.append(
            {
                "Contrast": row["contrast_id"],
                "Model": row["model_family"],
                "Event mean": fmt(row["event_mean_delta"]),
                "95% CI": f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]",
                "p": fmt_p(row["p_two_sided"]),
                "Excludes 0": "yes" if row["ci_excludes_zero"] else "no",
            }
        )

    t2_row = pooled_df[pooled_df["contrast_id"].eq("T2_minus_B0")].iloc[0]
    t3_row = pooled_df[pooled_df["contrast_id"].eq("T3_minus_B0")].iloc[0]
    b0_row = pooled_df[pooled_df["contrast_id"].eq("B0_minus_T1")].iloc[0]

    if bool(t2_row["ci_excludes_zero"]):
        claim = (
            "The strongest synthesis trade-off claim is supported: `T2 - B0` "
            "rationale diversity is positive and its event-level bootstrap 95% CI excludes zero."
        )
    elif bool(t3_row["ci_excludes_zero"]):
        claim = (
            "`T3 - B0` is significant while `T2 - B0` is directionally consistent but not "
            "separated from zero. The synthesis trade-off remains defensible, but the formal "
            "claim should be weaker for shared synthesis."
        )
    else:
        claim = (
            "Neither `T2 - B0` nor `T3 - B0` excludes zero in the event-level bootstrap. "
            "Use directional language only for the rationale mechanism."
        )

    lines = [
        "# B0 Event-Level Bootstrap Contrasts - 2026-05-10",
        "",
        "## Scope",
        "",
        "This memo runs event-cluster bootstrap inference for the B0 mechanism",
        "contrasts requested for the formal synthesis-tradeoff claim. The resampling",
        "unit is `event_id`: each bootstrap replicate samples the `94` events with",
        "replacement and recomputes the mean event-level rationale delta. Within each",
        "event, all `all_profiles` model/upstream cells are averaged before",
        "resampling.",
        "",
        f"- Input: `{contrast_csv.resolve()}`",
        f"- Metric: `{RATIONALE_METRIC}`",
        f"- Bootstrap reps: `{reps}`",
        f"- Seed: `{seed}`",
        "",
        "## Formal Readout",
        "",
        claim,
        "",
        md_table(
            display_rows,
            [
                ("Contrast", "Contrast"),
                ("Rows", "Rows"),
                ("Events", "Events"),
                ("Event mean", "Event mean"),
                ("Median", "Median"),
                ("Positive share", "Positive share"),
                ("95% CI", "95% CI"),
                ("p", "p"),
                ("Excludes 0", "Excludes 0"),
            ],
        ),
        "",
        "## Interpretation",
        "",
        f"- `B0 - T1` rationale delta is `{fmt(b0_row['event_mean_delta'])}` with CI "
        f"`[{fmt(b0_row['ci_low'])}, {fmt(b0_row['ci_high'])}]`.",
        f"- `T2 - B0` rationale delta is `{fmt(t2_row['event_mean_delta'])}` with CI "
        f"`[{fmt(t2_row['ci_low'])}, {fmt(t2_row['ci_high'])}]`.",
        f"- `T3 - B0` rationale delta is `{fmt(t3_row['event_mean_delta'])}` with CI "
        f"`[{fmt(t3_row['ci_low'])}, {fmt(t3_row['ci_high'])}]`.",
        "",
        "The paper-facing claim should use this event-level table rather than raw",
        "pooled row means, because events are the independent sampling unit.",
        "",
        "## Receiver-Model Diagnostic",
        "",
        md_table(
            by_model_rows,
            [
                ("Contrast", "Contrast"),
                ("Model", "Model"),
                ("Event mean", "Event mean"),
                ("95% CI", "95% CI"),
                ("p", "p"),
                ("Excludes 0", "Excludes 0"),
            ],
        ),
        "",
        "Generated tables:",
        "",
        f"- `{pooled_path.resolve()}`",
        f"- `{model_path.resolve()}`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    build_outputs(
        resolve(args.contrast_csv),
        resolve(args.output_dir),
        resolve(args.report_path),
        args.bootstrap_reps,
        args.seed,
    )


if __name__ == "__main__":
    main()
