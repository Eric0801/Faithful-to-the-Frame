#!/usr/bin/env python3
"""Freeze the R9AL-W3 conditional positive-prediction readout.

The readout uses the established four-family T1/B0 decision matrix. It groups
trace-level strictly-positive expected-return predictions by the realized
CAR[1,5] sign and writes both treatment-specific and pooled T1/B0 results.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "t4_followup_20260505"
    / "metrics_stage2_plus_t4_plus_b0_four_models_20260510"
    / "decision_rows.csv"
)
OUTPUT_DIR = ROOT / "docs" / "experiment_results" / "tables" / "rebuttal_w3_conditional_positive_predictions_20260712"
REPORT_PATH = ROOT / "docs" / "experiment_results" / "rebuttal_w3_conditional_positive_predictions_20260712.md"

TREATMENTS = ("B0_canonical_evidence_only", "T1_raw_public_information")
EXPECTED_ROWS_PER_EVENT = 24  # Four receiver models x six profiles.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def outcome_sign(car: float) -> str:
    if car > 0:
        return "positive_CAR"
    if car < 0:
        return "negative_CAR"
    return "zero_CAR"


def summarize(rows: pd.DataFrame, label: str) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for outcome in ("negative_CAR", "positive_CAR"):
        subset = rows.loc[rows["realized_outcome"] == outcome]
        summary.append(
            {
                "treatment_scope": label,
                "realized_outcome": outcome,
                "n_events": int(subset["event_id"].nunique()),
                "n_traces": int(len(subset)),
                "positive_prediction_rate": float(subset["predicted_positive"].mean()),
            }
        )
    negative_rate = summary[0]["positive_prediction_rate"]
    positive_rate = summary[1]["positive_prediction_rate"]
    for row in summary:
        row["positive_minus_negative_car_pp"] = float(100 * (positive_rate - negative_rate))
    return summary


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    rows = pd.read_csv(INPUT_PATH)
    required = {"event_id", "treatment", "expected_return_5d", "CAR_1_5"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = rows.loc[rows["treatment"].isin(TREATMENTS)].copy()
    if rows.empty:
        raise ValueError("No T1/B0 rows found.")
    if rows["expected_return_5d"].isna().any() or rows["CAR_1_5"].isna().any():
        raise ValueError("T1/B0 rows contain missing expected returns or CAR[1,5].")

    rows["realized_outcome"] = rows["CAR_1_5"].map(outcome_sign)
    if (rows["realized_outcome"] == "zero_CAR").any():
        raise ValueError("The frozen readout requires no exactly-zero CAR[1,5] events.")
    rows["predicted_positive"] = rows["expected_return_5d"] > 0

    events_per_treatment = rows.groupby("treatment")["event_id"].nunique()
    row_counts = rows.groupby("treatment").size()
    for treatment in TREATMENTS:
        if events_per_treatment.get(treatment) != 94:
            raise ValueError(f"{treatment} does not contain exactly 94 events.")
        if row_counts.get(treatment) != 94 * EXPECTED_ROWS_PER_EVENT:
            raise ValueError(f"{treatment} does not contain the expected decision rows.")

    summary_rows: list[dict[str, object]] = []
    for treatment in TREATMENTS:
        summary_rows.extend(summarize(rows.loc[rows["treatment"] == treatment], treatment))
    summary_rows.extend(summarize(rows, "pooled_T1_B0"))
    summary = pd.DataFrame(summary_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "conditional_positive_prediction_rates.csv"
    summary.to_csv(summary_path, index=False)

    pooled = summary.loc[summary["treatment_scope"] == "pooled_T1_B0"].set_index("realized_outcome")
    negative = pooled.loc["negative_CAR"]
    positive = pooled.loc["positive_CAR"]
    input_hash = sha256(INPUT_PATH)
    REPORT_PATH.write_text(
        "# R9AL-W3 Conditional Positive-Prediction Readout\n\n"
        "This frozen readout conditions T1/B0 trace-level strictly-positive "
        "five-day expected-return predictions on the realized CAR[1,5] sign. "
        "It uses no post-hoc filtering.\n\n"
        "## Frozen inputs\n\n"
        f"- Decision rows: `{INPUT_PATH.relative_to(ROOT)}`\n"
        f"- SHA-256: `{input_hash}`\n"
        f"- Treatments: `{TREATMENTS[0]}` and `{TREATMENTS[1]}`\n"
        "- Expected rows per treatment: 94 events x 4 models x 6 profiles = 2,256\n\n"
        "## Pooled T1/B0 result\n\n"
        "| Realized CAR[1,5] sign | Events | Traces | P+ | Positive-CAR minus negative-CAR |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
        f"| Negative | {int(negative['n_events'])} | {int(negative['n_traces'])} | "
        f"{100 * negative['positive_prediction_rate']:.2f}% | "
        f"{negative['positive_minus_negative_car_pp']:.2f} pp |\n"
        f"| Positive | {int(positive['n_events'])} | {int(positive['n_traces'])} | "
        f"{100 * positive['positive_prediction_rate']:.2f}% | "
        f"{positive['positive_minus_negative_car_pp']:.2f} pp |\n\n"
        "The corresponding treatment-specific values are in "
        "`conditional_positive_prediction_rates.csv`.\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
