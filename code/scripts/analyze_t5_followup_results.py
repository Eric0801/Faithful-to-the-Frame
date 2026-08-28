#!/usr/bin/env python3
"""Summarize the frozen T5/B0 policy and edit-bearing contrasts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_t5_edit_bearing_inference import (
    TREATMENTS,
    bootstrap_summary,
    exact_sign_test,
    paired_sign_flip_test,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "t5_linguistic_deframing_20260710"
)
DEFAULT_FREEZE_RECORD = DEFAULT_ROOT / "t5_manifest_freeze.json"
DEFAULT_REPS = 20_000
FULL_SIGN_FLIP_REPS = 100_000
TREATMENT_B0, TREATMENT_T5 = TREATMENTS
METRICS = (
    "positive_expected_return_rate",
    "directional_accuracy",
    "action_accuracy",
    "absolute_return_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-rows-csv", required=True)
    parser.add_argument("--freeze-record", default=str(DEFAULT_FREEZE_RECORD))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def metric_values(rows: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "positive_expected_return_rate":
        return (pd.to_numeric(rows["expected_return_5d"], errors="coerce") > 0).astype(float)
    return pd.to_numeric(rows[metric], errors="coerce")


def event_deltas(rows: pd.DataFrame, event_ids: list[str], metric: str) -> pd.DataFrame:
    subset = rows[
        rows["event_id"].astype(str).isin(event_ids)
        & rows["treatment"].isin((TREATMENT_B0, TREATMENT_T5))
    ].copy()
    subset["metric_value"] = metric_values(subset, metric)
    wide = (
        subset.groupby(["event_id", "treatment"], as_index=False)["metric_value"]
        .mean()
        .pivot(index="event_id", columns="treatment", values="metric_value")
    )
    missing = sorted(set(event_ids) - set(wide.dropna(subset=list(TREATMENTS)).index.astype(str)))
    if missing:
        raise ValueError(f"missing complete B0/T5 event pairs for {metric}: {missing}")
    paired = wide.loc[event_ids, list(TREATMENTS)].reset_index()
    paired["metric_name"] = metric
    paired["delta_t5_minus_b0"] = paired[TREATMENT_T5] - paired[TREATMENT_B0]
    return paired


def monte_carlo_sign_flip(deltas: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array((-1.0, 1.0)), size=(FULL_SIGN_FLIP_REPS, len(deltas)))
    simulated = (signs * deltas).mean(axis=1)
    return float((np.abs(simulated) >= abs(deltas.mean()) - 1e-15).mean())


def summarize(
    deltas: np.ndarray,
    *,
    metric: str,
    scope: str,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scope": scope,
        "metric": metric,
        "contrast": "T5_linguistic_deframing - B0_canonical_evidence_only",
        "n_events": int(len(deltas)),
        "mean_delta": float(deltas.mean()),
        "nonzero_event_deltas": int((np.abs(deltas) > 1e-12).sum()),
        "bootstrap": bootstrap_summary(deltas, reps, seed),
    }
    if metric == "positive_expected_return_rate":
        result["exact_sign_test"] = exact_sign_test(deltas)
        if len(deltas) <= 20:
            result["exact_paired_sign_flip_permutation"] = paired_sign_flip_test(deltas)
        else:
            result["monte_carlo_sign_flip_p_two_sided"] = monte_carlo_sign_flip(deltas, seed)
    return result


def main() -> int:
    args = parse_args()
    rows = pd.read_csv(resolve(args.decision_rows_csv), low_memory=False)
    freeze = json.loads(resolve(args.freeze_record).read_text(encoding="utf-8"))
    full_events = sorted(rows["event_id"].astype(str).unique())
    edit_events = [str(value) for value in freeze["edit_bearing_event_ids"]]
    scopes = {
        "dataset_wide_policy": full_events,
        "edit_bearing": edit_events,
    }

    summaries: list[dict[str, Any]] = []
    delta_frames: list[pd.DataFrame] = []
    for scope_index, (scope, event_ids) in enumerate(scopes.items()):
        for metric_index, metric in enumerate(METRICS):
            paired = event_deltas(rows, event_ids, metric)
            paired["scope"] = scope
            delta_frames.append(paired)
            summaries.append(
                summarize(
                    paired["delta_t5_minus_b0"].to_numpy(dtype=float),
                    metric=metric,
                    scope=scope,
                    reps=args.bootstrap_reps,
                    seed=args.seed + scope_index * 100 + metric_index,
                )
            )

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(delta_frames, ignore_index=True).to_csv(
        output_dir / "t5_followup_event_deltas.csv", index=False, lineterminator="\n"
    )
    payload = {
        "freeze_manifest_sha256": freeze["manifest_sha256"],
        "full_event_count": len(full_events),
        "edit_bearing_event_ids": edit_events,
        "summaries": summaries,
    }
    (output_dir / "t5_followup_summary.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
