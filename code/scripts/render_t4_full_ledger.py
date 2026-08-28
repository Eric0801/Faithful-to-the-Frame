#!/usr/bin/env python3
"""Render deterministic lossless T4 structured evidence ledger artifacts.

Main T4 is a rendering transformation, not an extraction or summarization step:
every canonical evidence unit is copied into the agent-facing ledger exactly
once. The audit sidecar verifies that the renderer introduces no representation
stage attrition relative to the canonical evidence bank.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_BANKS_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "evidence_banks_entity_visible"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "treatments_entity_visible"
    / "T4_full_structured_evidence_ledger"
)

TREATMENT_NAME = "T4_full_structured_evidence_ledger"
RENDERER_VERSION = "t4_full_ledger_v5_schema_once_pipe_ledger"
UNIT_FIELDS = (
    "evidence_id",
    "category",
    "claim",
    "value",
    "source_ids",
    "source_quote",
    "support_label",
)

POST_EVENT_MARKERS = (
    "post-event",
    "realized market reaction",
    "hidden label",
    "hidden outcome",
    "car_1_5",
    "actual return",
    "future filing",
    "future news",
)
RECOMMENDATION_RE = re.compile(
    r"\b(buy|sell|hold|recommend|recommendation|price target|upgrade|downgrade)\b",
    flags=re.IGNORECASE,
)
DIRECTIONAL_LABEL_RE = re.compile(
    r"\b(bullish|bearish|upside|downside|positive signal|negative catalyst|favorable setup)\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-banks-dir", default=str(DEFAULT_EVIDENCE_BANKS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--representation-seeds", default="1,2")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one representation seed is required")
    return seeds


def compact_whitespace(value: Any) -> str:
    return " ".join(str(value if value is not None else "").split())


def source_id_sort_key(source_id: Any) -> tuple[str, int, str]:
    text = str(source_id)
    prefix = "".join(ch for ch in text if not ch.isdigit())
    digits = "".join(ch for ch in text if ch.isdigit())
    return (prefix, int(digits or 0), text)


def evidence_id_sort_key(evidence_id: Any) -> tuple[str, int, str]:
    text = str(evidence_id)
    prefix = "".join(ch for ch in text if not ch.isdigit())
    digits = "".join(ch for ch in text if ch.isdigit())
    return (prefix, int(digits or 0), text)


def json_sha256(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def clean_source_ids(unit: dict[str, Any]) -> list[str]:
    return sorted(
        [str(item) for item in unit.get("source_ids", []) if str(item).strip()],
        key=source_id_sort_key,
    )


def canonical_unit(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": compact_whitespace(unit.get("evidence_id")),
        "category": compact_whitespace(unit.get("category") or "other"),
        "claim": compact_whitespace(unit.get("claim")),
        "value": unit.get("value"),
        "source_ids": clean_source_ids(unit),
        "source_quote": compact_whitespace(unit.get("source_quote")),
        "support_label": compact_whitespace(unit.get("support_label") or "supported"),
    }


def canonical_units(bank: dict[str, Any]) -> list[dict[str, Any]]:
    units = [canonical_unit(unit) for unit in bank.get("evidence_units", [])]
    return sorted(units, key=lambda unit: evidence_id_sort_key(unit["evidence_id"]))


def category_sort_key(category: str) -> tuple[int, str]:
    order = {
        "revenue": 0,
        "earnings_or_eps": 1,
        "margins": 2,
        "guidance": 3,
        "costs_or_expenses": 4,
        "risk_or_uncertainty": 5,
        "balance_sheet_or_cash": 6,
        "pre_event_price_context": 7,
        "other": 8,
    }
    return (order.get(category, 99), category)


def unit_to_text(unit: dict[str, Any]) -> list[str]:
    source_ids = ", ".join(unit["source_ids"])
    lines = [
        f"- evidence_id: {unit['evidence_id']}",
        f"  category: {unit['category']}",
        f"  claim: {unit['claim']}",
    ]
    if unit["value"] is not None and compact_whitespace(unit["value"]):
        lines.append(f"  value: {compact_whitespace(unit['value'])}")
    lines.extend(
        [
            f"  source_ids: [{source_ids}]",
            f"  support_label: {unit['support_label']}",
            f"  source_quote: {unit['source_quote']}",
        ]
    )
    return lines


def escape_markdown_table_cell(value: Any) -> str:
    return compact_whitespace(value).replace("\\", "\\\\").replace("|", "\\|")


def format_value(value: Any) -> str:
    text = compact_whitespace(value)
    return text


def format_source_quote(unit: dict[str, Any]) -> str:
    claim = compact_whitespace(unit["claim"])
    quote = compact_whitespace(unit["source_quote"])
    if quote == claim:
        return "SAME_AS_CLAIM"
    return quote


def render_bullet_text(bank: dict[str, Any], units: list[dict[str, Any]]) -> str:
    lines = [
        f"Entity: {company_label(bank)}",
        f"Event: {bank['event_id']}",
        "Representation: T4 full structured evidence ledger",
        "Design: deterministic lossless rendering of the full canonical evidence-unit bank.",
        "Rules: no evidence selection; no narrative synthesis; no action advice; no directional label.",
        f"Canonical evidence units included: {len(units)}",
    ]
    categories = sorted({unit["category"] for unit in units}, key=category_sort_key)
    for category in categories:
        lines.append(f"## {category}")
        for unit in [item for item in units if item["category"] == category]:
            lines.extend(unit_to_text(unit))
    lines.append("No action advice or directional label is added by the renderer.")
    return "\n".join(lines)


def company_label(bank: dict[str, Any]) -> str:
    name = compact_whitespace(bank.get("company_name") or bank.get("company_alias"))
    ticker = compact_whitespace(bank.get("ticker"))
    if name and ticker:
        return f"{name} ({ticker})"
    return name or ticker or compact_whitespace(bank.get("masked_company")) or "UNKNOWN_ENTITY"


def render_markdown_ledger_text(bank: dict[str, Any], units: list[dict[str, Any]]) -> str:
    all_supported = all(unit["support_label"] == "supported" for unit in units)
    lines = [
        f"Entity: {company_label(bank)}",
        f"Event: {bank['event_id']}",
        "Representation: T4 full structured evidence ledger",
        "Design: deterministic lossless rendering of the full canonical evidence-unit bank.",
        "Rules: no evidence selection; no narrative synthesis; no action advice; no directional label.",
        f"Canonical evidence units included: {len(units)}",
    ]
    if all_supported:
        lines.append("Support label default: supported for all rows.")
        lines.append("Columns: Evidence ID | Source IDs | Claim | Value | Source quote")
        row_separator = " | "
    else:
        lines.append("Columns: Evidence ID | Source IDs | Claim | Value | Support | Source quote")
        row_separator = " | "
    lines.append("Encoding: blank Value means NA; SAME_AS_CLAIM means source quote exactly equals claim.")
    categories = sorted({unit["category"] for unit in units}, key=category_sort_key)
    for category in categories:
        lines.extend(["", f"## Category: {category}"])
        for unit in [item for item in units if item["category"] == category]:
            row = [
                unit["evidence_id"],
                " ".join(unit["source_ids"]),
                unit["claim"],
                format_value(unit["value"]),
            ]
            if not all_supported:
                row.append(unit["support_label"])
            row.append(format_source_quote(unit))
            lines.append(row_separator.join(escape_markdown_table_cell(item) for item in row))
    lines.append("No action advice or directional label is added by the renderer.")
    return "\n".join(lines)


def render_text(bank: dict[str, Any], units: list[dict[str, Any]]) -> str:
    return render_markdown_ledger_text(bank, units)


def duplicate_ids(units: list[dict[str, Any]]) -> list[str]:
    counts = Counter(unit["evidence_id"] for unit in units)
    return sorted([evidence_id for evidence_id, count in counts.items() if count > 1], key=evidence_id_sort_key)


def compare_units(
    bank_units: list[dict[str, Any]],
    rendered_units: list[dict[str, Any]],
) -> dict[str, Any]:
    bank_by_id = {unit["evidence_id"]: unit for unit in bank_units}
    rendered_by_id = {unit["evidence_id"]: unit for unit in rendered_units}
    bank_ids = set(bank_by_id)
    rendered_ids = set(rendered_by_id)
    field_mismatches: list[dict[str, str]] = []
    for evidence_id in sorted(bank_ids & rendered_ids, key=evidence_id_sort_key):
        for field in UNIT_FIELDS:
            if bank_by_id[evidence_id].get(field) != rendered_by_id[evidence_id].get(field):
                field_mismatches.append(
                    {
                        "evidence_id": evidence_id,
                        "field": field,
                        "canonical_value": compact_whitespace(bank_by_id[evidence_id].get(field)),
                        "rendered_value": compact_whitespace(rendered_by_id[evidence_id].get(field)),
                    }
                )
    missing = sorted(bank_ids - rendered_ids, key=evidence_id_sort_key)
    extra = sorted(rendered_ids - bank_ids, key=evidence_id_sort_key)
    duplicate_rendered = duplicate_ids(rendered_units)
    return {
        "missing_evidence_ids": missing,
        "extra_evidence_ids": extra,
        "duplicate_evidence_ids": duplicate_rendered,
        "field_mismatches": field_mismatches,
        "lossless_pass": not missing and not extra and not duplicate_rendered and not field_mismatches,
    }


def marker_hits(rendered_text: str) -> dict[str, int]:
    lower_text = rendered_text.lower()
    return {
        "recommendation_terms": len(RECOMMENDATION_RE.findall(rendered_text)),
        "directional_labels": len(DIRECTIONAL_LABEL_RE.findall(rendered_text)),
        "post_event_marker_hits": sum(lower_text.count(marker) for marker in POST_EVENT_MARKERS),
    }


def payload_for_bank(
    bank: dict[str, Any],
    evidence_bank_path: Path,
    representation_seed: int,
) -> dict[str, Any]:
    units = canonical_units(bank)
    rendered_text = render_text(bank, units)
    return {
        "event_id": str(bank["event_id"]),
        "treatment": TREATMENT_NAME,
        "representation_seed": representation_seed,
        "renderer_version": RENDERER_VERSION,
        "serialization_format": "schema_once_pipe_ledger",
        "rendered_text": rendered_text,
        "evidence_bank_path": project_relative(evidence_bank_path),
        "canonical_evidence_unit_count": len(units),
        "structured_evidence_units": units,
        "structured_evidence_units_sha256": json_sha256(units),
    }


def audit_for_payload(
    bank: dict[str, Any],
    payload: dict[str, Any],
    evidence_bank_path: Path,
    payload_path: Path,
) -> dict[str, Any]:
    bank_units = canonical_units(bank)
    rendered_units = list(payload["structured_evidence_units"])
    comparison = compare_units(bank_units, rendered_units)
    hits = marker_hits(str(payload["rendered_text"]))
    previous_rendered_text = render_bullet_text(bank, bank_units)
    rendered_text = str(payload["rendered_text"])
    return {
        "event_id": str(bank["event_id"]),
        "treatment": TREATMENT_NAME,
        "representation_seed": payload["representation_seed"],
        "renderer_version": RENDERER_VERSION,
        "evidence_bank_path": project_relative(evidence_bank_path),
        "payload_path": project_relative(payload_path),
        "canonical_units_total": len(bank_units),
        "rendered_units_total": len(rendered_units),
        "canonical_evidence_ids_sha256": json_sha256([unit["evidence_id"] for unit in bank_units]),
        "rendered_evidence_ids_sha256": json_sha256([unit["evidence_id"] for unit in rendered_units]),
        "canonical_units_sha256": json_sha256(bank_units),
        "rendered_units_sha256": json_sha256(rendered_units),
        "serialization_format": payload.get("serialization_format", ""),
        "previous_bullet_text_chars": len(previous_rendered_text),
        "rendered_text_chars": len(rendered_text),
        "text_char_reduction_pct": round(
            100 * (len(previous_rendered_text) - len(rendered_text)) / len(previous_rendered_text),
            2,
        )
        if previous_rendered_text
        else 0,
        **comparison,
        **hits,
    }


def manifest_row(audit: dict[str, Any], audit_path: Path) -> dict[str, Any]:
    return {
        "event_id": audit["event_id"],
        "representation_seed": audit["representation_seed"],
        "canonical_units_total": audit["canonical_units_total"],
        "rendered_units_total": audit["rendered_units_total"],
        "lossless_pass": audit["lossless_pass"],
        "missing_evidence_ids": " ".join(audit["missing_evidence_ids"]),
        "extra_evidence_ids": " ".join(audit["extra_evidence_ids"]),
        "duplicate_evidence_ids": " ".join(audit["duplicate_evidence_ids"]),
        "field_mismatch_count": len(audit["field_mismatches"]),
        "serialization_format": audit["serialization_format"],
        "previous_bullet_text_chars": audit["previous_bullet_text_chars"],
        "rendered_text_chars": audit["rendered_text_chars"],
        "text_char_reduction_pct": audit["text_char_reduction_pct"],
        "recommendation_terms": audit["recommendation_terms"],
        "directional_labels": audit["directional_labels"],
        "post_event_marker_hits": audit["post_event_marker_hits"],
        "payload_path": audit["payload_path"],
        "audit_path": project_relative(audit_path),
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "event_id",
        "representation_seed",
        "canonical_units_total",
        "rendered_units_total",
        "lossless_pass",
        "missing_evidence_ids",
        "extra_evidence_ids",
        "duplicate_evidence_ids",
        "field_mismatch_count",
        "serialization_format",
        "previous_bullet_text_chars",
        "rendered_text_chars",
        "text_char_reduction_pct",
        "recommendation_terms",
        "directional_labels",
        "post_event_marker_hits",
        "payload_path",
        "audit_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    summary = {
        "treatment": TREATMENT_NAME,
        "renderer_version": RENDERER_VERSION,
        "events": len({row["event_id"] for row in rows}),
        "payloads": len(rows),
        "lossless_passed": sum(1 for row in rows if row["lossless_pass"] is True),
        "lossless_failed": sum(1 for row in rows if row["lossless_pass"] is not True),
        "max_rendered_text_chars": max((int(row["rendered_text_chars"]) for row in rows), default=0),
        "min_rendered_text_chars": min((int(row["rendered_text_chars"]) for row in rows), default=0),
        "mean_text_char_reduction_pct": round(
            sum(float(row["text_char_reduction_pct"]) for row in rows) / len(rows),
            2,
        )
        if rows
        else 0,
        "total_recommendation_terms": sum(int(row["recommendation_terms"]) for row in rows),
        "total_directional_labels": sum(int(row["directional_labels"]) for row in rows),
        "total_post_event_marker_hits": sum(int(row["post_event_marker_hits"]) for row in rows),
    }
    write_json(path, summary)


def main() -> None:
    args = parse_args()
    evidence_banks_dir = Path(args.evidence_banks_dir)
    output_dir = Path(args.output_dir)
    audit_dir = output_dir / "_audit"
    seeds = parse_seed_list(args.representation_seeds)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)

    evidence_bank_paths = sorted(evidence_banks_dir.glob("evt_*.json"))
    if not evidence_bank_paths:
        raise ValueError(f"no evt_*.json evidence banks found in {evidence_banks_dir}")

    manifest_rows: list[dict[str, Any]] = []
    for evidence_bank_path in evidence_bank_paths:
        bank = read_json(evidence_bank_path)
        event_id = str(bank["event_id"])
        for representation_seed in seeds:
            payload = payload_for_bank(bank, evidence_bank_path, representation_seed)
            payload_path = output_dir / f"{event_id}_rs{representation_seed}.json"
            audit_path = audit_dir / f"{event_id}_rs{representation_seed}.audit.json"
            write_json(payload_path, payload)
            audit = audit_for_payload(bank, payload, evidence_bank_path, payload_path)
            write_json(audit_path, audit)
            manifest_rows.append(manifest_row(audit, audit_path))

    write_manifest(audit_dir / "manifest.csv", manifest_rows)
    write_summary(audit_dir / "summary.json", manifest_rows)
    failures = [row for row in manifest_rows if row["lossless_pass"] is not True]
    print(f"T4 full ledger payloads: {len(manifest_rows)}")
    print(f"Events: {len(evidence_bank_paths)}")
    print(f"Lossless failures: {len(failures)}")
    print(f"Output: {output_dir}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
