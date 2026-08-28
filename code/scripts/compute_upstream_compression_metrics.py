#!/usr/bin/env python3
"""Compute upstream representation compression metrics for T2/T3 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_DIR = PROJECT_ROOT / "data" / "pilot_predata_2026_main"
DEFAULT_OUTPUT_DIR = DEFAULT_PILOT_DIR / "upstream_compression_20260503"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "experiment_results"
    / "pilot_upstream_compression_20260503.md"
)
SOURCE_ID_RE = re.compile(r"(?<![A-Za-z0-9])([SX]\d{3,4})(?![A-Za-z0-9])")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._%/-][A-Za-z0-9]+)*")

DEFAULT_REPRESENTATION_OUTPUTS = (
    DEFAULT_PILOT_DIR / "representation_outputs_20260430_pilot_gpt52.jsonl",
    DEFAULT_PILOT_DIR / "representation_outputs_20260430_pilot_claude45.jsonl",
    DEFAULT_PILOT_DIR / "representation_outputs_20260430_pilot_qwen3.jsonl",
    DEFAULT_PILOT_DIR / "representation_outputs_20260430_pilot_deepseek31.jsonl",
)

MODEL_LABELS_BY_FILENAME = {
    "gpt52": "gpt-5.2",
    "claude45": "claude-sonnet-4.5",
    "qwen3": "qwen3-235b-a22b",
    "deepseek31": "deepseek-v3.1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute upstream compression diagnostics from representation outputs."
    )
    parser.add_argument(
        "representation_outputs",
        nargs="*",
        help=(
            "Representation output JSONL files. Defaults to the four-model pilot "
            "outputs under data/pilot_predata_2026_main."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--near-zero-threshold",
        type=float,
        default=1e-12,
        help="Denominator threshold for normalized bottleneck indices.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is invalid JSONL") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(payload)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_list = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: serialize_value(row.get(field)) for field in field_list})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize_value(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return value
    return str(value)


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def source_ids_from_text(text: str) -> set[str]:
    return set(SOURCE_ID_RE.findall(text or ""))


def mean(values: Iterable[float | None]) -> float | None:
    numeric = [value for value in values if value is not None and math.isfinite(value)]
    if not numeric:
        return None
    return statistics.fmean(numeric)


def median(values: Iterable[float | None]) -> float | None:
    numeric = [value for value in values if value is not None and math.isfinite(value)]
    if not numeric:
        return None
    return statistics.median(numeric)


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def normalized_loss(reference: float | None, compressed: float | None, threshold: float) -> float | None:
    if reference is None or compressed is None:
        return None
    if abs(reference) <= threshold:
        return None
    return (reference - compressed) / reference


def support_share(values: Iterable[float | None]) -> float | None:
    numeric = [value for value in values if value is not None and math.isfinite(value)]
    if not numeric:
        return None
    return sum(1 for value in numeric if value > 0) / len(numeric)


def jaccard_distance(left: set[str], right: set[str]) -> float | None:
    union = left | right
    if not union:
        return None
    return 1.0 - (len(left & right) / len(union))


def mean_pairwise_jaccard_distance(sets: list[set[str]]) -> float | None:
    distances: list[float] = []
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            distance = jaccard_distance(left, right)
            if distance is not None:
                distances.append(distance)
    return mean(distances)


def infer_model_family(path: Path, row: dict[str, Any]) -> str:
    lower_name = path.name.lower()
    for key, label in MODEL_LABELS_BY_FILENAME.items():
        if key in lower_name:
            return label
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("model", "api_response_model"):
            value = metadata.get(key)
            if value:
                return str(value)
    value = row.get("upstream_model_family")
    return str(value) if value else "unknown_model"


def load_evidence_bank(path_text: str) -> dict[str, Any]:
    path = resolve_project_path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    units = payload.get("evidence_units")
    if not isinstance(units, list):
        raise ValueError(f"{path} has no evidence_units list")

    source_to_category: dict[str, str] = {}
    source_to_evidence_id: dict[str, str] = {}
    valid_source_ids: set[str] = set()
    bank_claim_tokens = 0
    source_backed_unit_count = 0
    categories: set[str] = set()

    for unit in units:
        if not isinstance(unit, dict):
            continue
        source_ids = [
            str(source_id)
            for source_id in unit.get("source_ids", [])
            if SOURCE_ID_RE.fullmatch(str(source_id))
        ]
        if not source_ids:
            continue
        category = str(unit.get("category") or "unknown")
        evidence_id = str(unit.get("evidence_id") or "")
        categories.add(category)
        source_backed_unit_count += 1
        bank_claim_tokens += token_count(str(unit.get("claim") or ""))
        for source_id in source_ids:
            valid_source_ids.add(source_id)
            source_to_category[source_id] = category
            source_to_evidence_id[source_id] = evidence_id

    return {
        "event_id": payload.get("event_id"),
        "ticker": payload.get("ticker"),
        "company_name": payload.get("company_name"),
        "market_cap_group": payload.get("market_cap_group"),
        "sector": payload.get("sector"),
        "event_date": payload.get("event_date"),
        "valid_source_ids": valid_source_ids,
        "source_to_category": source_to_category,
        "source_to_evidence_id": source_to_evidence_id,
        "source_count": len(valid_source_ids),
        "source_backed_unit_count": source_backed_unit_count,
        "category_count": len(categories),
        "bank_claim_tokens": bank_claim_tokens,
    }


def row_metrics(
    row: dict[str, Any],
    model_family: str,
    evidence_bank: dict[str, Any],
    source_path: Path,
    line_number: int,
) -> dict[str, Any]:
    rendered_text = str(row.get("rendered_text") or "")
    observed_source_ids = source_ids_from_text(rendered_text)
    valid_source_ids = set(evidence_bank["valid_source_ids"])
    cited_source_ids = observed_source_ids & valid_source_ids
    invalid_source_ids = observed_source_ids - valid_source_ids
    cited_evidence_ids = {
        evidence_bank["source_to_evidence_id"][source_id]
        for source_id in cited_source_ids
        if evidence_bank["source_to_evidence_id"].get(source_id)
    }
    cited_categories = {
        evidence_bank["source_to_category"][source_id]
        for source_id in cited_source_ids
        if evidence_bank["source_to_category"].get(source_id)
    }
    output_tokens = token_count(rendered_text)
    source_coverage = safe_ratio(len(cited_source_ids), evidence_bank["source_count"])
    evidence_unit_coverage = safe_ratio(
        len(cited_evidence_ids), evidence_bank["source_backed_unit_count"]
    )
    category_coverage = safe_ratio(len(cited_categories), evidence_bank["category_count"])
    output_to_bank_token_ratio = safe_ratio(output_tokens, evidence_bank["bank_claim_tokens"])

    return {
        "model_family": model_family,
        "event_id": row.get("event_id"),
        "ticker": evidence_bank.get("ticker"),
        "market_cap_group": evidence_bank.get("market_cap_group"),
        "sector": evidence_bank.get("sector"),
        "event_date": evidence_bank.get("event_date"),
        "treatment": row.get("treatment"),
        "representation_seed": row.get("representation_seed"),
        "profile_id": row.get("profile_id", "shared"),
        "generator_seed": row.get("generator_seed"),
        "source_path": project_relative(source_path),
        "line_number": line_number,
        "bank_source_count": evidence_bank["source_count"],
        "bank_evidence_unit_count": evidence_bank["source_backed_unit_count"],
        "bank_category_count": evidence_bank["category_count"],
        "bank_claim_tokens": evidence_bank["bank_claim_tokens"],
        "output_tokens": output_tokens,
        "output_to_bank_token_ratio": output_to_bank_token_ratio,
        "token_compression_ratio": (
            None if output_to_bank_token_ratio is None else 1.0 - output_to_bank_token_ratio
        ),
        "cited_source_count": len(cited_source_ids),
        "observed_source_ref_count": len(observed_source_ids),
        "invalid_source_ref_count": len(invalid_source_ids),
        "cited_evidence_unit_count": len(cited_evidence_ids),
        "cited_category_count": len(cited_categories),
        "source_coverage": source_coverage,
        "evidence_unit_coverage": evidence_unit_coverage,
        "category_coverage": category_coverage,
        "cited_source_ids": ";".join(sorted(cited_source_ids)),
        "invalid_source_ids": ";".join(sorted(invalid_source_ids)),
        "cited_categories": ";".join(sorted(cited_categories)),
    }


def group_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (
            str(row["model_family"]),
            str(row["event_id"]),
            str(row["representation_seed"]),
        )
        grouped[key][str(row["treatment"])].append(row)
    return grouped


def set_from_semicolon(text: Any) -> set[str]:
    if not text:
        return set()
    return {part for part in str(text).split(";") if part}


def aggregate_cell_metrics(
    row_metric_rows: list[dict[str, Any]], near_zero_threshold: float
) -> list[dict[str, Any]]:
    cell_rows: list[dict[str, Any]] = []
    for (model_family, event_id, representation_seed), by_treatment in sorted(
        group_rows(row_metric_rows).items()
    ):
        t2_rows = by_treatment.get("T2_shared_summary", [])
        t3_rows = by_treatment.get("T3_independent_summary", [])
        if not t2_rows or not t3_rows:
            continue
        if len(t2_rows) != 1:
            raise ValueError(
                f"Expected one T2 row for {model_family}/{event_id}/seed {representation_seed}; "
                f"found {len(t2_rows)}"
            )
        t2 = t2_rows[0]
        t2_sources = set_from_semicolon(t2["cited_source_ids"])
        t2_categories = set_from_semicolon(t2["cited_categories"])
        t3_source_sets = [set_from_semicolon(row["cited_source_ids"]) for row in t3_rows]
        t3_category_sets = [set_from_semicolon(row["cited_categories"]) for row in t3_rows]
        t3_union_sources = set().union(*t3_source_sets)
        t3_union_categories = set().union(*t3_category_sets)

        bank_source_count = int(t2["bank_source_count"])
        bank_category_count = int(t2["bank_category_count"])
        t3_union_source_coverage = safe_ratio(len(t3_union_sources), bank_source_count)
        t3_union_category_coverage = safe_ratio(len(t3_union_categories), bank_category_count)
        t2_source_coverage = t2["source_coverage"]
        t2_category_coverage = t2["category_coverage"]
        t3_mean_source_coverage = mean(row["source_coverage"] for row in t3_rows)
        t3_mean_category_coverage = mean(row["category_coverage"] for row in t3_rows)

        row = {
            "model_family": model_family,
            "event_id": event_id,
            "representation_seed": representation_seed,
            "ticker": t2.get("ticker"),
            "market_cap_group": t2.get("market_cap_group"),
            "sector": t2.get("sector"),
            "event_date": t2.get("event_date"),
            "t3_profile_count": len(t3_rows),
            "bank_source_count": bank_source_count,
            "bank_category_count": bank_category_count,
            "bank_claim_tokens": t2.get("bank_claim_tokens"),
            "t2_output_tokens": t2.get("output_tokens"),
            "t3_mean_output_tokens": mean(row["output_tokens"] for row in t3_rows),
            "t2_token_compression_ratio": t2.get("token_compression_ratio"),
            "t3_mean_token_compression_ratio": mean(
                row["token_compression_ratio"] for row in t3_rows
            ),
            "t2_source_coverage": t2_source_coverage,
            "t3_mean_source_coverage": t3_mean_source_coverage,
            "t3_union_source_coverage": t3_union_source_coverage,
            "t2_category_coverage": t2_category_coverage,
            "t3_mean_category_coverage": t3_mean_category_coverage,
            "t3_union_category_coverage": t3_union_category_coverage,
            "t3_pairwise_source_jaccard_distance": mean_pairwise_jaccard_distance(
                t3_source_sets
            ),
            "t3_pairwise_category_jaccard_distance": mean_pairwise_jaccard_distance(
                t3_category_sets
            ),
            "t2_vs_t3_union_source_rbi": normalized_loss(
                t3_union_source_coverage, t2_source_coverage, near_zero_threshold
            ),
            "t2_vs_t3_union_category_rbi": normalized_loss(
                t3_union_category_coverage, t2_category_coverage, near_zero_threshold
            ),
            "t2_vs_t3_mean_source_delta": (
                None
                if t3_mean_source_coverage is None or t2_source_coverage is None
                else t3_mean_source_coverage - t2_source_coverage
            ),
            "t2_vs_t3_mean_category_delta": (
                None
                if t3_mean_category_coverage is None or t2_category_coverage is None
                else t3_mean_category_coverage - t2_category_coverage
            ),
        }
        cell_rows.append(row)
    return cell_rows


def summarize_by_group(
    rows: list[dict[str, Any]], group_field: str, metric_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field, ""))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for group_value, group_rows_for_value in sorted(grouped.items()):
        summary: dict[str, Any] = {
            group_field: group_value,
            "n_cells": len(group_rows_for_value),
        }
        for metric in metric_fields:
            values = [row.get(metric) for row in group_rows_for_value]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_median"] = median(values)
            summary[f"{metric}_positive_share"] = support_share(values)
        summary_rows.append(summary)
    return summary_rows


def build_run_summary(
    representation_paths: list[Path],
    row_metric_rows: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
    model_summary_rows: list[dict[str, Any]],
    market_cap_summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "created_at_utc": utc_now_iso(),
        "representation_outputs": [project_relative(path) for path in representation_paths],
        "row_metrics_count": len(row_metric_rows),
        "paired_cell_count": len(cell_rows),
        "model_count": len({row["model_family"] for row in row_metric_rows}),
        "event_count": len({row["event_id"] for row in row_metric_rows}),
        "treatment_counts": {
            treatment: sum(1 for row in row_metric_rows if row["treatment"] == treatment)
            for treatment in sorted({row["treatment"] for row in row_metric_rows})
        },
        "overall": {
            "t2_source_coverage_mean": mean(row["t2_source_coverage"] for row in cell_rows),
            "t2_source_coverage_median": median(row["t2_source_coverage"] for row in cell_rows),
            "t2_source_coverage_positive_share": support_share(
                row["t2_source_coverage"] for row in cell_rows
            ),
            "t3_mean_source_coverage_mean": mean(
                row["t3_mean_source_coverage"] for row in cell_rows
            ),
            "t3_mean_source_coverage_median": median(
                row["t3_mean_source_coverage"] for row in cell_rows
            ),
            "t3_mean_source_coverage_positive_share": support_share(
                row["t3_mean_source_coverage"] for row in cell_rows
            ),
            "t2_vs_t3_mean_source_delta_mean": mean(
                row["t2_vs_t3_mean_source_delta"] for row in cell_rows
            ),
            "t2_vs_t3_mean_source_delta_median": median(
                row["t2_vs_t3_mean_source_delta"] for row in cell_rows
            ),
            "t2_vs_t3_mean_source_delta_positive_share": support_share(
                row["t2_vs_t3_mean_source_delta"] for row in cell_rows
            ),
            "t3_union_source_coverage_mean": mean(
                row["t3_union_source_coverage"] for row in cell_rows
            ),
            "t2_vs_t3_union_source_rbi_mean": mean(
                row["t2_vs_t3_union_source_rbi"] for row in cell_rows
            ),
            "t2_vs_t3_union_source_rbi_median": median(
                row["t2_vs_t3_union_source_rbi"] for row in cell_rows
            ),
            "t2_vs_t3_union_source_rbi_positive_share": support_share(
                row["t2_vs_t3_union_source_rbi"] for row in cell_rows
            ),
            "t2_category_coverage_mean": mean(row["t2_category_coverage"] for row in cell_rows),
            "t2_category_coverage_median": median(
                row["t2_category_coverage"] for row in cell_rows
            ),
            "t2_category_coverage_positive_share": support_share(
                row["t2_category_coverage"] for row in cell_rows
            ),
            "t3_mean_category_coverage_mean": mean(
                row["t3_mean_category_coverage"] for row in cell_rows
            ),
            "t3_mean_category_coverage_median": median(
                row["t3_mean_category_coverage"] for row in cell_rows
            ),
            "t3_mean_category_coverage_positive_share": support_share(
                row["t3_mean_category_coverage"] for row in cell_rows
            ),
            "t2_vs_t3_mean_category_delta_mean": mean(
                row["t2_vs_t3_mean_category_delta"] for row in cell_rows
            ),
            "t2_vs_t3_mean_category_delta_median": median(
                row["t2_vs_t3_mean_category_delta"] for row in cell_rows
            ),
            "t2_vs_t3_mean_category_delta_positive_share": support_share(
                row["t2_vs_t3_mean_category_delta"] for row in cell_rows
            ),
            "t3_union_category_coverage_mean": mean(
                row["t3_union_category_coverage"] for row in cell_rows
            ),
            "t2_vs_t3_union_category_rbi_mean": mean(
                row["t2_vs_t3_union_category_rbi"] for row in cell_rows
            ),
            "t2_vs_t3_union_category_rbi_median": median(
                row["t2_vs_t3_union_category_rbi"] for row in cell_rows
            ),
            "t2_vs_t3_union_category_rbi_positive_share": support_share(
                row["t2_vs_t3_union_category_rbi"] for row in cell_rows
            ),
            "t2_token_compression_ratio_mean": mean(
                row["t2_token_compression_ratio"] for row in cell_rows
            ),
            "t2_token_compression_ratio_median": median(
                row["t2_token_compression_ratio"] for row in cell_rows
            ),
            "t2_token_compression_ratio_positive_share": support_share(
                row["t2_token_compression_ratio"] for row in cell_rows
            ),
            "t3_mean_token_compression_ratio_mean": mean(
                row["t3_mean_token_compression_ratio"] for row in cell_rows
            ),
            "t3_mean_token_compression_ratio_median": median(
                row["t3_mean_token_compression_ratio"] for row in cell_rows
            ),
            "t3_mean_token_compression_ratio_positive_share": support_share(
                row["t3_mean_token_compression_ratio"] for row in cell_rows
            ),
            "t3_pairwise_source_jaccard_distance_mean": mean(
                row["t3_pairwise_source_jaccard_distance"] for row in cell_rows
            ),
            "t3_pairwise_source_jaccard_distance_median": median(
                row["t3_pairwise_source_jaccard_distance"] for row in cell_rows
            ),
            "t3_pairwise_source_jaccard_distance_positive_share": support_share(
                row["t3_pairwise_source_jaccard_distance"] for row in cell_rows
            ),
            "t3_pairwise_category_jaccard_distance_mean": mean(
                row["t3_pairwise_category_jaccard_distance"] for row in cell_rows
            ),
            "t3_pairwise_category_jaccard_distance_median": median(
                row["t3_pairwise_category_jaccard_distance"] for row in cell_rows
            ),
            "t3_pairwise_category_jaccard_distance_positive_share": support_share(
                row["t3_pairwise_category_jaccard_distance"] for row in cell_rows
            ),
        },
        "model_summary": model_summary_rows,
        "market_cap_summary": market_cap_summary_rows,
    }


def format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], fields: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(format_number(row.get(field)) for _, field in fields)
            + " |"
        )
    return "\n".join([header, separator, *body])


def write_report(
    path: Path,
    run_summary: dict[str, Any],
    model_summary_rows: list[dict[str, Any]],
    market_cap_summary_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    overall = run_summary["overall"]
    report = f"""# Pilot Upstream Compression Diagnostic - 2026-05-03

Purpose: estimate Stage 1 representation compression before spending provider
budget on the official main run.

## Scope

- Input: four-model pilot T2/T3 representation outputs.
- Events: `{run_summary['event_count']}`
- Models: `{run_summary['model_count']}`
- Row metrics: `{run_summary['row_metrics_count']}`
- Paired model-event-seed cells: `{run_summary['paired_cell_count']}`
- Output directory: `{project_relative(output_dir)}`

## Correction

The T3-union coverage index is useful only as an exploratory upper-bound
coverage diagnostic. It should not be used as the primary Stage 1 metric because
it compares six independent T3 summaries against one T2 shared summary, so the
larger T3 source universe is partly mechanical.

The Stage 1 construct should instead separate:

- absolute compression against the canonical evidence bank;
- one-to-one T2 vs mean T3 coverage differences;
- within-T3 representation diversity across independent summaries;
- optional interpreted-frame diversity, if a separate extraction/judge step is
  added.

## Revised Definitions

- `source_coverage`: cited valid `S###`/`X###` source IDs divided by all source
  IDs in the canonical evidence bank.
- `category_coverage`: cited evidence categories divided by all source-backed
  categories in the canonical evidence bank.
- `token_compression_ratio`: `1 - output_tokens / bank_claim_tokens`, using a
  deterministic lexical token count.
- `T2_vs_T3_mean_source_delta`: mean single-summary T3 source coverage minus T2
  source coverage. This is a fairer one-to-one coverage comparison than T3
  union coverage.
- `T3_pairwise_source_jaccard_distance`: average pairwise source-set distance
  among the six independent T3 summaries for the same event, model, and seed.
  This measures whether T3 actually creates diverse evidence paths.

## Revised Overall Result

| Metric | Mean | Median | Positive share |
| --- | ---: | ---: | ---: |
| T2 source coverage | {format_number(overall['t2_source_coverage_mean'])} | {format_number(overall['t2_source_coverage_median'])} | {format_number(overall['t2_source_coverage_positive_share'])} |
| T3 mean single-summary source coverage | {format_number(overall['t3_mean_source_coverage_mean'])} | {format_number(overall['t3_mean_source_coverage_median'])} | {format_number(overall['t3_mean_source_coverage_positive_share'])} |
| T3 mean minus T2 source coverage | {format_number(overall['t2_vs_t3_mean_source_delta_mean'])} | {format_number(overall['t2_vs_t3_mean_source_delta_median'])} | {format_number(overall['t2_vs_t3_mean_source_delta_positive_share'])} |
| T2 category coverage | {format_number(overall['t2_category_coverage_mean'])} | {format_number(overall['t2_category_coverage_median'])} | {format_number(overall['t2_category_coverage_positive_share'])} |
| T3 mean single-summary category coverage | {format_number(overall['t3_mean_category_coverage_mean'])} | {format_number(overall['t3_mean_category_coverage_median'])} | {format_number(overall['t3_mean_category_coverage_positive_share'])} |
| T3 mean minus T2 category coverage | {format_number(overall['t2_vs_t3_mean_category_delta_mean'])} | {format_number(overall['t2_vs_t3_mean_category_delta_median'])} | {format_number(overall['t2_vs_t3_mean_category_delta_positive_share'])} |
| T3 pairwise source Jaccard distance | {format_number(overall['t3_pairwise_source_jaccard_distance_mean'])} | {format_number(overall['t3_pairwise_source_jaccard_distance_median'])} | {format_number(overall['t3_pairwise_source_jaccard_distance_positive_share'])} |
| T3 pairwise category Jaccard distance | {format_number(overall['t3_pairwise_category_jaccard_distance_mean'])} | {format_number(overall['t3_pairwise_category_jaccard_distance_median'])} | {format_number(overall['t3_pairwise_category_jaccard_distance_positive_share'])} |
| T2 token compression | {format_number(overall['t2_token_compression_ratio_mean'])} | {format_number(overall['t2_token_compression_ratio_median'])} | {format_number(overall['t2_token_compression_ratio_positive_share'])} |
| T3 mean token compression | {format_number(overall['t3_mean_token_compression_ratio_mean'])} | {format_number(overall['t3_mean_token_compression_ratio_median'])} | {format_number(overall['t3_mean_token_compression_ratio_positive_share'])} |

## By Model

{markdown_table(model_summary_rows, [
    ('Model', 'model_family'),
    ('n', 'n_cells'),
    ('T3-T2 source cov delta', 't2_vs_t3_mean_source_delta_mean'),
    ('T3 source diversity', 't3_pairwise_source_jaccard_distance_mean'),
    ('T3-T2 category cov delta', 't2_vs_t3_mean_category_delta_mean'),
    ('T3 category diversity', 't3_pairwise_category_jaccard_distance_mean'),
])}

## By Market Cap Group

{markdown_table(market_cap_summary_rows, [
    ('Market cap', 'market_cap_group'),
    ('n', 'n_cells'),
    ('T3-T2 source cov delta', 't2_vs_t3_mean_source_delta_mean'),
    ('T3 source diversity', 't3_pairwise_source_jaccard_distance_mean'),
    ('T3-T2 category cov delta', 't2_vs_t3_mean_category_delta_mean'),
    ('T3 category diversity', 't3_pairwise_category_jaccard_distance_mean'),
])}

## Deprecated Union-Coverage Diagnostic

The old union diagnostic was:

- T2 source coverage mean: `{format_number(overall['t2_source_coverage_mean'])}`
- T3 union source coverage mean: `{format_number(overall['t3_union_source_coverage_mean'])}`
- normalized union gap: `{format_number(overall['t2_vs_t3_union_source_rbi_mean'])}`

This should be reported, if at all, as a descriptive upper-bound contrast:
T3's independent-summary pool spans a larger combined evidence menu than one
shared T2 summary. It should not be called the primary representation
bottleneck metric.

## Revised Readout

Pilot upstream results do not support the strong claim that a single T2 summary
has dramatically lower per-summary source coverage than a single T3 summary.
The fair one-to-one source coverage gap is small:
`{format_number(overall['t2_vs_t3_mean_source_delta_mean'])}`.

The stronger Stage 1 signal is different: T3 independent summaries create
multiple distinct evidence paths. The average within-T3 source-set Jaccard
distance is `{format_number(overall['t3_pairwise_source_jaccard_distance_mean'])}`,
and it is positive in
`{format_number(overall['t3_pairwise_source_jaccard_distance_positive_share'])}`
of model-event-seed cells. This supports a better framing for Stage 1: shared
summaries remove recipient-level interpretive variation by giving everyone the
same evidence path, while independent summaries preserve room for different
evidence paths.

The pilot result should not be used as a final claim because it has only 12
events and includes provider-era artifacts. It is useful as a metric sanity
check and visualization rehearsal for the official Stage 1 audit.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    representation_paths = [
        resolve_project_path(path)
        for path in (args.representation_outputs or DEFAULT_REPRESENTATION_OUTPUTS)
    ]
    output_dir = resolve_project_path(args.output_dir)
    report_path = resolve_project_path(args.report_path)

    evidence_cache: dict[str, dict[str, Any]] = {}
    row_metric_rows: list[dict[str, Any]] = []

    for path in representation_paths:
        rows = read_jsonl(path)
        for line_number, row in enumerate(rows, start=1):
            evidence_bank_path = str(row.get("evidence_bank_path") or "")
            if not evidence_bank_path:
                raise ValueError(f"{path}:{line_number} has no evidence_bank_path")
            if evidence_bank_path not in evidence_cache:
                evidence_cache[evidence_bank_path] = load_evidence_bank(evidence_bank_path)
            model_family = infer_model_family(path, row)
            row_metric_rows.append(
                row_metrics(
                    row=row,
                    model_family=model_family,
                    evidence_bank=evidence_cache[evidence_bank_path],
                    source_path=path,
                    line_number=line_number,
                )
            )

    cell_rows = aggregate_cell_metrics(
        row_metric_rows=row_metric_rows,
        near_zero_threshold=args.near_zero_threshold,
    )
    metric_fields = (
        "t2_source_coverage",
        "t3_mean_source_coverage",
        "t2_vs_t3_mean_source_delta",
        "t3_union_source_coverage",
        "t2_vs_t3_union_source_rbi",
        "t2_category_coverage",
        "t3_mean_category_coverage",
        "t2_vs_t3_mean_category_delta",
        "t3_union_category_coverage",
        "t2_vs_t3_union_category_rbi",
        "t2_token_compression_ratio",
        "t3_mean_token_compression_ratio",
        "t3_pairwise_source_jaccard_distance",
        "t3_pairwise_category_jaccard_distance",
    )
    model_summary_rows = summarize_by_group(cell_rows, "model_family", metric_fields)
    market_cap_summary_rows = summarize_by_group(cell_rows, "market_cap_group", metric_fields)
    run_summary = build_run_summary(
        representation_paths=representation_paths,
        row_metric_rows=row_metric_rows,
        cell_rows=cell_rows,
        model_summary_rows=model_summary_rows,
        market_cap_summary_rows=market_cap_summary_rows,
    )

    row_fields = (
        "model_family",
        "event_id",
        "ticker",
        "market_cap_group",
        "sector",
        "event_date",
        "treatment",
        "representation_seed",
        "profile_id",
        "generator_seed",
        "source_path",
        "line_number",
        "bank_source_count",
        "bank_evidence_unit_count",
        "bank_category_count",
        "bank_claim_tokens",
        "output_tokens",
        "output_to_bank_token_ratio",
        "token_compression_ratio",
        "cited_source_count",
        "observed_source_ref_count",
        "invalid_source_ref_count",
        "cited_evidence_unit_count",
        "cited_category_count",
        "source_coverage",
        "evidence_unit_coverage",
        "category_coverage",
        "cited_source_ids",
        "invalid_source_ids",
        "cited_categories",
    )
    cell_fields = (
        "model_family",
        "event_id",
        "representation_seed",
        "ticker",
        "market_cap_group",
        "sector",
        "event_date",
        "t3_profile_count",
        "bank_source_count",
        "bank_category_count",
        "bank_claim_tokens",
        "t2_output_tokens",
        "t3_mean_output_tokens",
        "t2_token_compression_ratio",
        "t3_mean_token_compression_ratio",
        "t2_source_coverage",
        "t3_mean_source_coverage",
        "t3_union_source_coverage",
        "t2_category_coverage",
        "t3_mean_category_coverage",
        "t3_union_category_coverage",
        "t3_pairwise_source_jaccard_distance",
        "t3_pairwise_category_jaccard_distance",
        "t2_vs_t3_union_source_rbi",
        "t2_vs_t3_union_category_rbi",
        "t2_vs_t3_mean_source_delta",
        "t2_vs_t3_mean_category_delta",
    )

    summary_fields = ["n_cells"]
    for metric in metric_fields:
        summary_fields.extend(
            [f"{metric}_mean", f"{metric}_median", f"{metric}_positive_share"]
        )

    write_csv(output_dir / "row_metrics.csv", row_metric_rows, row_fields)
    write_csv(output_dir / "cell_metrics.csv", cell_rows, cell_fields)
    write_csv(
        output_dir / "model_summary.csv",
        model_summary_rows,
        ["model_family", *summary_fields],
    )
    write_csv(
        output_dir / "market_cap_summary.csv",
        market_cap_summary_rows,
        ["market_cap_group", *summary_fields],
    )
    write_json(output_dir / "run_summary.json", run_summary)
    write_report(
        path=report_path,
        run_summary=run_summary,
        model_summary_rows=model_summary_rows,
        market_cap_summary_rows=market_cap_summary_rows,
        output_dir=output_dir,
    )
    return run_summary


def main() -> int:
    args = parse_args()
    summary = run(args)
    overall = summary["overall"]
    print("upstream compression summary:")
    print(f"  row_metrics_count: {summary['row_metrics_count']}")
    print(f"  paired_cell_count: {summary['paired_cell_count']}")
    print(
        "  t2_vs_t3_mean_source_delta_mean: "
        f"{format_number(overall['t2_vs_t3_mean_source_delta_mean'])}"
    )
    print(
        "  t3_pairwise_source_jaccard_distance_mean: "
        f"{format_number(overall['t3_pairwise_source_jaccard_distance_mean'])}"
    )
    print(
        "  t3_pairwise_source_jaccard_distance_positive_share: "
        f"{format_number(overall['t3_pairwise_source_jaccard_distance_positive_share'])}"
    )
    print(
        "  t2_token_compression_ratio_mean: "
        f"{format_number(overall['t2_token_compression_ratio_mean'])}"
    )
    print(
        "  t3_mean_token_compression_ratio_mean: "
        f"{format_number(overall['t3_mean_token_compression_ratio_mean'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
