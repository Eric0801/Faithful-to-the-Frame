#!/usr/bin/env python3
"""Run pre-specified event-level T5-minus-B0 inference for the frozen E set."""

from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "t5_linguistic_deframing_20260710"
)
DEFAULT_FREEZE_RECORD = DEFAULT_ROOT / "t5_manifest_freeze.json"
DEFAULT_REPS = 20_000
TREATMENTS = ("B0_canonical_evidence_only", "T5_linguistic_deframing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-rows-csv", required=True)
    parser.add_argument("--freeze-record", default=str(DEFAULT_FREEZE_RECORD))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--metric",
        choices=("positive_expected_return_rate", "mean_expected_return_5d", "buy_rate"),
        default="positive_expected_return_rate",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def metric_values(rows: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "positive_expected_return_rate":
        return (pd.to_numeric(rows["expected_return_5d"], errors="coerce") > 0).astype(float)
    if metric == "mean_expected_return_5d":
        return pd.to_numeric(rows["expected_return_5d"], errors="coerce")
    if metric == "buy_rate":
        return rows["action"].astype(str).str.lower().eq("buy").astype(float)
    raise ValueError(f"unsupported metric: {metric}")


def exact_sign_test(deltas: np.ndarray) -> dict[str, Any]:
    nonzero = deltas[np.abs(deltas) > 1e-12]
    n = int(len(nonzero))
    positives = int((nonzero > 0).sum())
    if n == 0:
        return {"n_nonzero": 0, "positive_events": 0, "two_sided_p": math.nan}
    lower_tail = sum(math.comb(n, k) for k in range(positives + 1)) / (2**n)
    upper_tail = sum(math.comb(n, k) for k in range(positives, n + 1)) / (2**n)
    return {
        "n_nonzero": n,
        "positive_events": positives,
        "two_sided_p": min(1.0, 2 * min(lower_tail, upper_tail)),
    }


def paired_sign_flip_test(deltas: np.ndarray) -> dict[str, Any]:
    observed = float(deltas.mean())
    n = len(deltas)
    if n > 20:
        raise ValueError("exact sign-flip test is limited to 20 events")
    statistics = np.fromiter(
        (np.mean(np.asarray(signs, dtype=float) * deltas) for signs in product((-1, 1), repeat=n)),
        dtype=float,
        count=2**n,
    )
    p_value = float((np.abs(statistics) >= abs(observed) - 1e-15).mean())
    return {"statistic": observed, "permutations": int(len(statistics)), "two_sided_p": p_value}


def bootstrap_summary(deltas: np.ndarray, reps: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(deltas, size=(reps, len(deltas)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"mean_delta": float(deltas.mean()), "ci_low": float(low), "ci_high": float(high)}


def paired_event_deltas(rows: pd.DataFrame, event_ids: list[str], metric: str) -> pd.DataFrame:
    required = {"event_id", "treatment", "expected_return_5d", "action"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"decision rows missing columns: {sorted(missing)}")
    subset = rows[rows["event_id"].astype(str).isin(event_ids)].copy()
    subset = subset[subset["treatment"].isin(TREATMENTS)].copy()
    subset["metric_value"] = metric_values(subset, metric)
    event_treatment = subset.groupby(["event_id", "treatment"], as_index=False)["metric_value"].mean()
    wide = event_treatment.pivot(index="event_id", columns="treatment", values="metric_value")
    missing_pairs = sorted(set(event_ids) - set(wide.dropna(subset=list(TREATMENTS)).index.astype(str)))
    if missing_pairs:
        raise ValueError(f"missing complete B0/T5 pairs for events: {missing_pairs}")
    paired = wide.loc[event_ids, list(TREATMENTS)].reset_index()
    paired["delta_t5_minus_b0"] = paired[TREATMENTS[1]] - paired[TREATMENTS[0]]
    return paired


def main() -> int:
    args = parse_args()
    freeze_record = json.loads(resolve(args.freeze_record).read_text(encoding="utf-8"))
    event_ids = [str(value) for value in freeze_record["edit_bearing_event_ids"]]
    rows = pd.read_csv(resolve(args.decision_rows_csv), low_memory=False)
    paired = paired_event_deltas(rows, event_ids, args.metric)
    deltas = paired["delta_t5_minus_b0"].to_numpy(dtype=float)
    summary = {
        "analysis_scope": "pre-specified edit-bearing event effect",
        "metric": args.metric,
        "contrast": "T5_linguistic_deframing - B0_canonical_evidence_only",
        "event_ids": event_ids,
        "n_events": len(event_ids),
        "bootstrap": bootstrap_summary(deltas, args.bootstrap_reps, args.seed),
        "exact_sign_test": exact_sign_test(deltas),
        "exact_paired_sign_flip_permutation": paired_sign_flip_test(deltas),
    }
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / "t5_edit_bearing_event_deltas.csv", index=False, lineterminator="\n")
    (output_dir / "t5_edit_bearing_inference.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
