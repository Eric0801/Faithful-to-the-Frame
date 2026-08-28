#!/usr/bin/env python3
"""Compare neutral-style and no-style T2/T3 metric bundles side by side."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CORE_OUTPUTS = {
    "cell_metrics": "cell_metrics.csv",
    "decision_rows": "decision_rows.csv",
    "t2_t3_contrasts": "t2_t3_contrasts.csv",
    "treatment_contrasts": "treatment_contrasts.csv",
    "run_summary": "run_summary.json",
}

CELL_GROUP_FIELDS = (
    "treatment_family",
    "treatment",
    "investor_group",
    "upstream_model_family",
    "model_family",
)
T2_T3_GROUP_FIELDS = (
    "investor_group",
    "upstream_model_family",
    "model_family",
)
TREATMENT_CONTRAST_GROUP_FIELDS = (
    "contrast_id",
    "baseline_treatment_family",
    "comparison_treatment_family",
    "metric_name",
    "metric_group",
    "metric_unit",
    "investor_group",
    "upstream_model_family",
    "model_family",
)
SOURCE_UPTAKE_GROUP_FIELDS = (
    "treatment_family",
    "treatment",
    "investor_group",
    "upstream_model_family",
    "model_family",
)
QUALITY_ACTION_GROUP_FIELDS = (
    "treatment_family",
    "treatment",
    "investor_group",
    "upstream_model_family",
    "model_family",
)

CELL_EXCLUDE_NUMERIC_FIELDS = {
    "representation_seed",
    "decision_seed",
    "profile_count",
    "expected_profile_count",
    "pair_count",
    "CAR_1_5",
}
T2_T3_MEAN_FIELDS = (
    "delta_std_expected_return_5d",
    "sci_std_expected_return_5d",
    "delta_mean_pairwise_abs_diff_expected_return_5d",
    "sci_mean_pairwise_abs_diff_expected_return_5d",
    "delta_iqr_expected_return_5d",
    "sci_iqr_expected_return_5d",
    "delta_confidence_dispersion",
    "sci_confidence_dispersion",
    "delta_entropy_buy_hold_sell",
    "sci_entropy_buy_hold_sell",
    "delta_action_diversity_one_minus_hhi",
    "sci_action_diversity_one_minus_hhi",
    "delta_action_direction_diversity_one_minus_abs_net_direction",
    "sci_action_direction_diversity_one_minus_abs_net_direction",
    "delta_mean_pairwise_sentence_embedding_distance",
    "sci_mean_pairwise_sentence_embedding_distance",
    "delta_source_id_diversity_one_minus_overlap",
    "sci_source_id_diversity_one_minus_overlap",
    "delta_evidence_category_diversity_one_minus_overlap",
    "sci_evidence_category_diversity_one_minus_overlap",
)
TREATMENT_CONTRAST_MEAN_FIELDS = ("delta", "normalized_index")
QUALITY_ACTION_MEAN_FIELDS = (
    "expected_return_5d",
    "confidence",
    "action_code",
    "action_strength",
    "directional_accuracy",
    "action_accuracy",
    "signed_return_error",
    "absolute_return_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare old neutral-style and new no-style T2/T3 metric bundles. "
            "Missing optional files and columns are skipped."
        )
    )
    parser.add_argument("--old-label", required=True)
    parser.add_argument("--old-metrics-dir", required=True, type=Path)
    parser.add_argument("--new-label", required=True)
    parser.add_argument("--new-metrics-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--old-stage1-rows", type=Path, default=None)
    parser.add_argument("--new-stage1-rows", type=Path, default=None)
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {"value": value}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.12g}"
    if value is None:
        return ""
    return value


def key_for(row: dict[str, str], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in fields)


def present_fields(fields: tuple[str, ...], fieldnames: list[str]) -> tuple[str, ...]:
    available = set(fieldnames)
    return tuple(field for field in fields if field in available)


def infer_numeric_fields(
    rows: list[dict[str, str]], fieldnames: list[str], exclude: set[str]
) -> list[str]:
    numeric_fields: list[str] = []
    for field in fieldnames:
        if field in exclude:
            continue
        values = [parse_float(row.get(field)) for row in rows if row.get(field, "").strip()]
        if values and any(value is not None for value in values):
            numeric_fields.append(field)
    return numeric_fields


def aggregate_means(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    group_fields: tuple[str, ...],
    value_fields: list[str],
) -> dict[tuple[str, ...], dict[str, Any]]:
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    actual_group_fields = present_fields(group_fields, fieldnames)
    for row in rows:
        key = key_for(row, actual_group_fields)
        if key not in groups:
            groups[key] = {
                "_group_fields": actual_group_fields,
                "row_count": 0,
                "values": {field: [] for field in value_fields},
            }
            for index, field in enumerate(actual_group_fields):
                groups[key][field] = key[index]
        groups[key]["row_count"] += 1
        for field in value_fields:
            value = parse_float(row.get(field))
            if value is not None:
                groups[key]["values"][field].append(value)

    aggregated: dict[tuple[str, ...], dict[str, Any]] = {}
    for key, group in groups.items():
        output = {
            field: group.get(field, "")
            for field in group["_group_fields"]
        }
        output["row_count"] = group["row_count"]
        for field in value_fields:
            output[field] = mean(group["values"][field])
        aggregated[key] = output
    return aggregated


def merge_side_by_side(
    old_label: str,
    old_rows: dict[tuple[str, ...], dict[str, Any]],
    new_label: str,
    new_rows: dict[tuple[str, ...], dict[str, Any]],
    group_fields: tuple[str, ...],
    value_fields: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    output_rows: list[dict[str, Any]] = []
    actual_group_fields = tuple(
        field
        for field in group_fields
        if any(field in row for row in old_rows.values())
        or any(field in row for row in new_rows.values())
    )
    all_keys = sorted(set(old_rows) | set(new_rows))
    for key in all_keys:
        old_row = old_rows.get(key, {})
        new_row = new_rows.get(key, {})
        output: dict[str, Any] = {}
        for field in actual_group_fields:
            output[field] = old_row.get(field, new_row.get(field, ""))
        output[f"{old_label}_row_count"] = old_row.get("row_count", "")
        output[f"{new_label}_row_count"] = new_row.get("row_count", "")
        for field in value_fields:
            old_value = old_row.get(field)
            new_value = new_row.get(field)
            output[f"{old_label}_{field}"] = format_value(old_value)
            output[f"{new_label}_{field}"] = format_value(new_value)
            output[f"delta_{new_label}_minus_{old_label}_{field}"] = format_value(
                new_value - old_value
                if isinstance(old_value, float) and isinstance(new_value, float)
                else None
            )
        output_rows.append(output)

    fieldnames = (
        list(actual_group_fields)
        + [f"{old_label}_row_count", f"{new_label}_row_count"]
        + [
            column
            for field in value_fields
            for column in (
                f"{old_label}_{field}",
                f"{new_label}_{field}",
                f"delta_{new_label}_minus_{old_label}_{field}",
            )
        ]
    )
    return output_rows, fieldnames


def parse_json_list(value: str | None) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def source_uptake_rows(
    rows: list[dict[str, str]], fieldnames: list[str]
) -> tuple[dict[tuple[str, ...], dict[str, Any]], list[str]]:
    if "evidence_used" not in fieldnames:
        return {}, []
    actual_group_fields = present_fields(SOURCE_UPTAKE_GROUP_FIELDS, fieldnames)
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = key_for(row, actual_group_fields)
        if key not in groups:
            groups[key] = {
                "_group_fields": actual_group_fields,
                "row_count": 0,
                "source_counts": [],
                "unique_sources": set(),
            }
            for index, field in enumerate(actual_group_fields):
                groups[key][field] = key[index]
        sources = [str(source) for source in parse_json_list(row.get("evidence_used"))]
        groups[key]["row_count"] += 1
        groups[key]["source_counts"].append(float(len(sources)))
        groups[key]["unique_sources"].update(sources)

    output: dict[tuple[str, ...], dict[str, Any]] = {}
    for key, group in groups.items():
        row = {field: group.get(field, "") for field in group["_group_fields"]}
        row["row_count"] = group["row_count"]
        row["mean_evidence_sources_per_decision"] = mean(group["source_counts"])
        row["unique_evidence_sources"] = float(len(group["unique_sources"]))
        output[key] = row
    return output, ["mean_evidence_sources_per_decision", "unique_evidence_sources"]


def bool_share_rows(
    rows: list[dict[str, str]], group_fields: tuple[str, ...], fieldnames: list[str]
) -> dict[tuple[str, ...], dict[str, float]]:
    if "action" not in fieldnames:
        return {}
    actual_group_fields = present_fields(group_fields, fieldnames)
    counts: dict[tuple[str, ...], defaultdict[str, int]] = {}
    for row in rows:
        key = key_for(row, actual_group_fields)
        counts.setdefault(key, defaultdict(int))
        action = row.get("action", "").strip().lower()
        if action:
            counts[key][f"action_share_{action}"] += 1
            counts[key]["action_total"] += 1
    output: dict[tuple[str, ...], dict[str, float]] = {}
    for key, counter in counts.items():
        total = counter.get("action_total", 0)
        if not total:
            continue
        output[key] = {
            field: counter.get(field, 0) / total
            for field in ("action_share_buy", "action_share_hold", "action_share_sell")
        }
    return output


def quality_action_rows(
    rows: list[dict[str, str]], fieldnames: list[str]
) -> tuple[dict[tuple[str, ...], dict[str, Any]], list[str]]:
    value_fields = [
        field for field in QUALITY_ACTION_MEAN_FIELDS if field in fieldnames
    ]
    if not value_fields and "action" not in fieldnames:
        return {}, []
    aggregated = aggregate_means(
        rows, fieldnames, QUALITY_ACTION_GROUP_FIELDS, value_fields
    )
    action_shares = bool_share_rows(rows, QUALITY_ACTION_GROUP_FIELDS, fieldnames)
    action_fields = ["action_share_buy", "action_share_hold", "action_share_sell"]
    for key, shares in action_shares.items():
        aggregated.setdefault(key, {"row_count": 0})
        aggregated[key].update(shares)
    return aggregated, value_fields + action_fields


def compare_table(
    old_label: str,
    old_rows: list[dict[str, str]],
    old_fieldnames: list[str],
    new_label: str,
    new_rows: list[dict[str, str]],
    new_fieldnames: list[str],
    group_fields: tuple[str, ...],
    value_fields: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    old_agg = aggregate_means(old_rows, old_fieldnames, group_fields, value_fields)
    new_agg = aggregate_means(new_rows, new_fieldnames, group_fields, value_fields)
    return merge_side_by_side(
        old_label, old_agg, new_label, new_agg, group_fields, value_fields
    )


def stage1_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False}
    rows, fieldnames = read_csv_rows(path)
    return {
        "provided": True,
        "path": str(path),
        "exists": path.exists(),
        "row_count": len(rows),
        "columns": fieldnames,
    }


def run() -> int:
    args = parse_args()
    old_dir = args.old_metrics_dir
    new_dir = args.new_metrics_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    old_cell_rows, old_cell_fields = read_csv_rows(old_dir / CORE_OUTPUTS["cell_metrics"])
    new_cell_rows, new_cell_fields = read_csv_rows(new_dir / CORE_OUTPUTS["cell_metrics"])
    old_t2_t3_rows, old_t2_t3_fields = read_csv_rows(
        old_dir / CORE_OUTPUTS["t2_t3_contrasts"]
    )
    new_t2_t3_rows, new_t2_t3_fields = read_csv_rows(
        new_dir / CORE_OUTPUTS["t2_t3_contrasts"]
    )
    old_contrast_rows, old_contrast_fields = read_csv_rows(
        old_dir / CORE_OUTPUTS["treatment_contrasts"]
    )
    new_contrast_rows, new_contrast_fields = read_csv_rows(
        new_dir / CORE_OUTPUTS["treatment_contrasts"]
    )
    old_decision_rows, old_decision_fields = read_csv_rows(
        old_dir / CORE_OUTPUTS["decision_rows"]
    )
    new_decision_rows, new_decision_fields = read_csv_rows(
        new_dir / CORE_OUTPUTS["decision_rows"]
    )

    emitted: list[str] = []
    skipped: dict[str, str] = {}

    cell_value_fields = sorted(
        set(infer_numeric_fields(old_cell_rows, old_cell_fields, CELL_EXCLUDE_NUMERIC_FIELDS))
        | set(infer_numeric_fields(new_cell_rows, new_cell_fields, CELL_EXCLUDE_NUMERIC_FIELDS))
    )
    if old_cell_rows or new_cell_rows:
        rows, fields = compare_table(
            args.old_label,
            old_cell_rows,
            old_cell_fields,
            args.new_label,
            new_cell_rows,
            new_cell_fields,
            CELL_GROUP_FIELDS,
            cell_value_fields,
        )
        write_csv(output_dir / "treatment_means.csv", rows, fields)
        emitted.append("treatment_means.csv")
    else:
        skipped["treatment_means.csv"] = "cell_metrics.csv missing in both bundles"

    t2_t3_value_fields = [
        field
        for field in T2_T3_MEAN_FIELDS
        if field in old_t2_t3_fields or field in new_t2_t3_fields
    ]
    if old_t2_t3_rows or new_t2_t3_rows:
        rows, fields = compare_table(
            args.old_label,
            old_t2_t3_rows,
            old_t2_t3_fields,
            args.new_label,
            new_t2_t3_rows,
            new_t2_t3_fields,
            T2_T3_GROUP_FIELDS,
            t2_t3_value_fields,
        )
        write_csv(output_dir / "t3_t2_pairwise_contrasts.csv", rows, fields)
        emitted.append("t3_t2_pairwise_contrasts.csv")
    else:
        skipped["t3_t2_pairwise_contrasts.csv"] = (
            "t2_t3_contrasts.csv missing in both bundles"
        )

    treatment_value_fields = [
        field
        for field in TREATMENT_CONTRAST_MEAN_FIELDS
        if field in old_contrast_fields or field in new_contrast_fields
    ]
    if old_contrast_rows or new_contrast_rows:
        rows, fields = compare_table(
            args.old_label,
            old_contrast_rows,
            old_contrast_fields,
            args.new_label,
            new_contrast_rows,
            new_contrast_fields,
            TREATMENT_CONTRAST_GROUP_FIELDS,
            treatment_value_fields,
        )
        write_csv(output_dir / "treatment_contrasts.csv", rows, fields)
        emitted.append("treatment_contrasts.csv")
    else:
        skipped["treatment_contrasts.csv"] = (
            "treatment_contrasts.csv missing in both bundles"
        )

    old_source_agg, source_fields = source_uptake_rows(
        old_decision_rows, old_decision_fields
    )
    new_source_agg, new_source_fields = source_uptake_rows(
        new_decision_rows, new_decision_fields
    )
    source_fields = sorted(set(source_fields) | set(new_source_fields))
    if old_source_agg or new_source_agg:
        rows, fields = merge_side_by_side(
            args.old_label,
            old_source_agg,
            args.new_label,
            new_source_agg,
            SOURCE_UPTAKE_GROUP_FIELDS,
            source_fields,
        )
        write_csv(output_dir / "final_source_uptake.csv", rows, fields)
        emitted.append("final_source_uptake.csv")
    else:
        skipped["final_source_uptake.csv"] = (
            "decision_rows.csv missing or lacks evidence_used"
        )

    old_quality_agg, quality_fields = quality_action_rows(
        old_decision_rows, old_decision_fields
    )
    new_quality_agg, new_quality_fields = quality_action_rows(
        new_decision_rows, new_decision_fields
    )
    quality_fields = [
        field
        for field in QUALITY_ACTION_MEAN_FIELDS
        if field in set(quality_fields) | set(new_quality_fields)
    ] + [
        field
        for field in ("action_share_buy", "action_share_hold", "action_share_sell")
        if field in set(quality_fields) | set(new_quality_fields)
    ]
    if old_quality_agg or new_quality_agg:
        rows, fields = merge_side_by_side(
            args.old_label,
            old_quality_agg,
            args.new_label,
            new_quality_agg,
            QUALITY_ACTION_GROUP_FIELDS,
            quality_fields,
        )
        write_csv(output_dir / "quality_action_summary.csv", rows, fields)
        emitted.append("quality_action_summary.csv")
    else:
        skipped["quality_action_summary.csv"] = (
            "decision_rows.csv missing or lacks action/quality columns"
        )

    old_summary = read_json(old_dir / CORE_OUTPUTS["run_summary"])
    new_summary = read_json(new_dir / CORE_OUTPUTS["run_summary"])
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "old_label": args.old_label,
        "old_metrics_dir": str(old_dir),
        "new_label": args.new_label,
        "new_metrics_dir": str(new_dir),
        "output_dir": str(output_dir),
        "emitted_files": emitted,
        "skipped_outputs": skipped,
        "input_row_counts": {
            args.old_label: {
                "cell_metrics": len(old_cell_rows),
                "decision_rows": len(old_decision_rows),
                "t2_t3_contrasts": len(old_t2_t3_rows),
                "treatment_contrasts": len(old_contrast_rows),
            },
            args.new_label: {
                "cell_metrics": len(new_cell_rows),
                "decision_rows": len(new_decision_rows),
                "t2_t3_contrasts": len(new_t2_t3_rows),
                "treatment_contrasts": len(new_contrast_rows),
            },
        },
        "input_columns": {
            args.old_label: {
                "cell_metrics": old_cell_fields,
                "decision_rows": old_decision_fields,
                "t2_t3_contrasts": old_t2_t3_fields,
                "treatment_contrasts": old_contrast_fields,
            },
            args.new_label: {
                "cell_metrics": new_cell_fields,
                "decision_rows": new_decision_fields,
                "t2_t3_contrasts": new_t2_t3_fields,
                "treatment_contrasts": new_contrast_fields,
            },
        },
        "stage1_rows": {
            args.old_label: stage1_summary(args.old_stage1_rows),
            args.new_label: stage1_summary(args.new_stage1_rows),
        },
        "source_run_summaries": {
            args.old_label: old_summary,
            args.new_label: new_summary,
        },
    }
    write_json(output_dir / "run_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
