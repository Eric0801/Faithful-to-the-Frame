#!/usr/bin/env python3
"""Reference E2A metric computation for the paper artifact release."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


ACTION_CODES = {"buy": 1, "hold": 0, "sell": -1}
TREATMENT_ORDER = ("T1", "B0", "T2", "T3", "T4")
CONTRASTS = (
    ("T2_minus_T1", "T2", "T1"),
    ("T3_minus_T2", "T3", "T2"),
    ("T4_minus_T2", "T4", "T2"),
    ("T2_minus_B0", "T2", "B0"),
    ("T3_minus_B0", "T3", "B0"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision_rows_csv", type=Path)
    parser.add_argument("--hidden-outcomes-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return sum(clean) / len(clean) if clean else None


def population_std(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return None
    center = sum(clean) / len(clean)
    return math.sqrt(sum((value - center) ** 2 for value in clean) / len(clean))


def pairwise_abs_diff(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if len(clean) < 2:
        return None
    return mean(abs(a - b) for a, b in combinations(clean, 2))


def parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [item.strip() for item in re.split(r"[;,|]", text) if item.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return []


def action_entropy(actions: list[str]) -> float | None:
    clean = [action.lower() for action in actions if action.lower() in ACTION_CODES]
    if not clean:
        return None
    counts = Counter(clean)
    total = len(clean)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def action_hhi(actions: list[str]) -> float | None:
    clean = [action.lower() for action in actions if action.lower() in ACTION_CODES]
    if not clean:
        return None
    counts = Counter(clean)
    total = len(clean)
    return sum((count / total) ** 2 for count in counts.values())


def pairwise_overlap_diversity(items: list[list[str]]) -> float | None:
    sets = [set(item for item in row if item) for row in items]
    if len(sets) < 2:
        return None
    overlaps: list[float] = []
    for left, right in combinations(sets, 2):
        union = left | right
        if not union:
            overlaps.append(1.0)
        else:
            overlaps.append(len(left & right) / len(union))
    overlap = mean(overlaps)
    return None if overlap is None else 1.0 - overlap


def tokenize(text: str) -> Counter[str]:
    return Counter(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def cosine_distance(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 1.0
    dot = sum(left[token] * right.get(token, 0) for token in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    return 1.0 - dot / (left_norm * right_norm)


def rationale_distance(texts: list[str]) -> float | None:
    clean = [text for text in texts if text.strip()]
    if len(clean) < 2:
        return None
    distances = [
        cosine_distance(tokenize(left), tokenize(right))
        for left, right in combinations(clean, 2)
    ]
    return mean(distances)


def row_treatment_family(row: dict[str, str]) -> str:
    family = row.get("treatment_family", "").strip()
    if family:
        return family
    treatment = row.get("treatment", "").strip()
    if treatment.startswith("B0"):
        return "B0"
    return treatment.split("_", 1)[0]


def row_rationale(row: dict[str, str]) -> str:
    for field in ("rationale", "reasoning", "key_reasons"):
        value = row.get(field, "")
        if field == "key_reasons":
            return " ".join(parse_list(value))
        if value.strip():
            return value
    return ""


def group_decisions(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        event_id = row.get("event_id", "").strip()
        treatment = row_treatment_family(row)
        if not event_id or not treatment:
            continue
        groups.setdefault((event_id, treatment), []).append(row)
    return groups


def event_metric_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (event_id, treatment), group in sorted(group_decisions(rows).items()):
        expected_returns = [as_float(row.get("expected_return_5d")) for row in group]
        confidences = [as_float(row.get("confidence")) for row in group]
        actions = [row.get("action", "").strip().lower() for row in group]
        source_lists = [parse_list(row.get("evidence_used")) for row in group]
        category_lists = [parse_list(row.get("evidence_categories")) for row in group]
        rationales = [row_rationale(row) for row in group]
        hhi = action_hhi(actions)
        output.append(
            {
                "event_id": event_id,
                "treatment_family": treatment,
                "decision_count": len(group),
                "std_expected_return_5d": population_std(expected_returns),
                "mean_pairwise_abs_diff_expected_return_5d": pairwise_abs_diff(expected_returns),
                "confidence_dispersion": population_std(confidences),
                "entropy_buy_hold_sell": action_entropy(actions),
                "action_diversity_one_minus_hhi": None if hhi is None else 1.0 - hhi,
                "mean_pairwise_sentence_embedding_distance": rationale_distance(rationales),
                "source_id_diversity_one_minus_overlap": pairwise_overlap_diversity(source_lists),
                "evidence_category_diversity_one_minus_overlap": pairwise_overlap_diversity(category_lists),
            }
        )
    return output


def treatment_means(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = [
        key for key in event_rows[0] if key not in {"event_id", "treatment_family", "decision_count"}
    ] if event_rows else []
    for treatment in TREATMENT_ORDER:
        group = [row for row in event_rows if row["treatment_family"] == treatment]
        if not group:
            continue
        item: dict[str, Any] = {
            "treatment_family": treatment,
            "n_events": len({row["event_id"] for row in group}),
            "n_event_treatment_cells": len(group),
        }
        for metric in metrics:
            item[metric] = mean(as_float(row.get(metric)) for row in group)
        output.append(item)
    return output


def paired_contrasts(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        key for key in event_rows[0] if key not in {"event_id", "treatment_family", "decision_count"}
    ] if event_rows else []
    by_event = {
        (row["event_id"], row["treatment_family"]): row
        for row in event_rows
    }
    event_ids = sorted({row["event_id"] for row in event_rows})
    output: list[dict[str, Any]] = []
    for contrast_id, comparison, baseline in CONTRASTS:
        for metric in metrics:
            deltas: list[float] = []
            for event_id in event_ids:
                left = by_event.get((event_id, comparison), {}).get(metric)
                right = by_event.get((event_id, baseline), {}).get(metric)
                left_value = as_float(left)
                right_value = as_float(right)
                if left_value is not None and right_value is not None:
                    deltas.append(left_value - right_value)
            output.append(
                {
                    "contrast_id": contrast_id,
                    "comparison_treatment_family": comparison,
                    "baseline_treatment_family": baseline,
                    "metric_name": metric,
                    "n_events": len(deltas),
                    "mean_delta": mean(deltas),
                    "positive_share": mean(1.0 if delta > 0 else 0.0 for delta in deltas),
                }
            )
    return output


def load_hidden_outcomes(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    outcomes: dict[str, float] = {}
    for row in read_csv(path):
        car = as_float(row.get("CAR_1_5"))
        event_id = row.get("event_id", "").strip()
        if event_id and car is not None:
            outcomes[event_id] = car
    return outcomes


def sign(value: float | None) -> int:
    if value is None:
        return 0
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def quality_rows(rows: list[dict[str, str]], hidden: dict[str, float]) -> list[dict[str, Any]]:
    if not hidden:
        return []
    groups = group_decisions(rows)
    output: list[dict[str, Any]] = []
    for (event_id, treatment), group in sorted(groups.items()):
        car = hidden.get(event_id)
        if car is None:
            continue
        directional = []
        strict = []
        abs_errors = []
        for row in group:
            expected = as_float(row.get("expected_return_5d"))
            action = row.get("action", "").strip().lower()
            directional.append(1.0 if sign(expected) == sign(car) else 0.0)
            strict.append(1.0 if ACTION_CODES.get(action, 0) == sign(car) else 0.0)
            if expected is not None:
                abs_errors.append(abs(expected - car))
        output.append(
            {
                "event_id": event_id,
                "treatment_family": treatment,
                "directional_accuracy": mean(directional),
                "strict_action_accuracy": mean(strict),
                "absolute_return_error": mean(abs_errors),
            }
        )
    return output


def main() -> int:
    args = parse_args()
    decisions = read_csv(args.decision_rows_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    events = event_metric_rows(decisions)
    means = treatment_means(events)
    contrasts = paired_contrasts(events)
    hidden = load_hidden_outcomes(args.hidden_outcomes_csv)
    quality = quality_rows(decisions, hidden)
    quality_means = treatment_means(quality) if quality else []

    event_fields = list(events[0].keys()) if events else []
    mean_fields = list(means[0].keys()) if means else []
    contrast_fields = list(contrasts[0].keys()) if contrasts else []
    quality_fields = list(quality_means[0].keys()) if quality_means else []
    write_csv(args.output_dir / "event_metrics.csv", events, event_fields)
    write_csv(args.output_dir / "treatment_means.csv", means, mean_fields)
    write_csv(args.output_dir / "paired_contrasts.csv", contrasts, contrast_fields)
    if quality_means:
        write_csv(args.output_dir / "quality_by_treatment.csv", quality_means, quality_fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
