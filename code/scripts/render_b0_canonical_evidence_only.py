#!/usr/bin/env python3
"""Render B0 canonical-evidence-only downstream payloads.

B0 is a deterministic baseline for the T2/T3/T4 canonical-evidence substrate:
it exposes the same canonical evidence-unit bank serialization used by the
actual T2/T3 upstream prompt jobs, but sends that substrate directly to the
downstream decision agent without LLM synthesis or T4 ledger formatting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
    / "b0_canonical_baseline_20260510"
    / "b0_payloads_event_level"
)

TREATMENT_NAME = "B0_canonical_evidence_only"
UPSTREAM_MODEL_FAMILY = "deterministic_canonical_evidence"
RENDERER_VERSION = "b0_canonical_evidence_only_v1_t2_t3_bank_jsonl"
SERIALIZATION_FORMAT = "t2_t3_upstream_canonical_bank_jsonl_no_source_quote"
CANONICAL_BANK_FIELDS = (
    "evidence_id",
    "category",
    "claim",
    "value",
    "source_ids",
    "support_label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-banks-dir", default=str(DEFAULT_EVIDENCE_BANKS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--representation-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def json_sha256(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compact_whitespace(value: Any) -> str:
    return " ".join(str(value if value is not None else "").split())


def event_metadata_lines(bank: dict[str, Any]) -> list[str]:
    metadata = [
        ("Entity", bank.get("company_name") or bank.get("company_alias") or bank.get("masked_company")),
        ("Ticker", bank.get("ticker")),
        ("Event date", bank.get("event_date")),
        ("Sector", bank.get("sector")),
        ("Market cap group", bank.get("market_cap_group")),
        ("Fiscal period", bank.get("fiscal_period")),
    ]
    return [
        f"- {label}: {compact_whitespace(value)}"
        for label, value in metadata
        if compact_whitespace(value)
    ]


def canonical_bank_unit(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": unit.get("evidence_id"),
        "category": unit.get("category"),
        "claim": unit.get("claim"),
        "value": unit.get("value"),
        "source_ids": unit.get("source_ids"),
        "support_label": unit.get("support_label"),
    }


def canonical_bank_units(bank: dict[str, Any]) -> list[dict[str, Any]]:
    units = bank.get("evidence_units")
    if not isinstance(units, list):
        raise ValueError(f"{bank.get('event_id', '<unknown>')} evidence_units must be a list")
    return [canonical_bank_unit(unit) for unit in units if isinstance(unit, dict)]


def serialize_evidence_bank_units(units: list[dict[str, Any]]) -> str:
    return "\n".join(
        json.dumps(unit, ensure_ascii=True, sort_keys=True)
        for unit in units
    )


def render_text(bank: dict[str, Any], units: list[dict[str, Any]]) -> str:
    return (
        "Event metadata:\n"
        + "\n".join(event_metadata_lines(bank))
        + "\n\n"
        + "Canonical evidence-unit bank:\n"
        + serialize_evidence_bank_units(units)
    )


def validate_units(units: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    seen_ids: set[str] = set()
    for index, unit in enumerate(units, start=1):
        missing = [field for field in CANONICAL_BANK_FIELDS if field not in unit]
        if missing:
            warnings.append(f"unit {index} missing fields: {', '.join(missing)}")
        evidence_id = compact_whitespace(unit.get("evidence_id"))
        if not evidence_id:
            warnings.append(f"unit {index} missing evidence_id")
        elif evidence_id in seen_ids:
            warnings.append(f"duplicate evidence_id: {evidence_id}")
        seen_ids.add(evidence_id)
        source_ids = unit.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids:
            warnings.append(f"{evidence_id or f'unit {index}'} missing source_ids")
    return warnings


def payload_for_bank(
    bank: dict[str, Any],
    evidence_bank_path: Path,
    representation_seed: int,
) -> dict[str, Any]:
    units = canonical_bank_units(bank)
    rendered_text = render_text(bank, units)
    warnings = validate_units(units)
    return {
        "event_id": str(bank["event_id"]),
        "treatment": TREATMENT_NAME,
        "upstream_model_family": UPSTREAM_MODEL_FAMILY,
        "representation_seed": representation_seed,
        "renderer_version": RENDERER_VERSION,
        "serialization_format": SERIALIZATION_FORMAT,
        "rendered_text": rendered_text,
        "evidence_bank_path": project_relative(evidence_bank_path),
        "source_packet_path": bank.get("source_packet_path"),
        "canonical_evidence_unit_count": len(units),
        "canonical_bank_units_sha256": json_sha256(units),
        "rendered_text_sha256": hashlib.sha256(rendered_text.encode("utf-8")).hexdigest(),
        "validation_warnings": warnings,
    }


def main() -> int:
    args = parse_args()
    evidence_banks_dir = Path(args.evidence_banks_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    for bank_path in sorted(evidence_banks_dir.glob("evt_*.json")):
        bank = read_json(bank_path)
        payload = payload_for_bank(bank, bank_path, args.representation_seed)
        payload_path = output_dir / f"{payload['event_id']}_rs{args.representation_seed}.json"
        write_json(payload_path, payload)
        payloads.append(payload)
        for warning in payload["validation_warnings"]:
            warning_rows.append({"event_id": payload["event_id"], "warning": warning})

    if not payloads:
        raise ValueError(f"no evt_*.json evidence banks found in {evidence_banks_dir}")

    summary = {
        "treatment": TREATMENT_NAME,
        "upstream_model_family": UPSTREAM_MODEL_FAMILY,
        "renderer_version": RENDERER_VERSION,
        "serialization_format": SERIALIZATION_FORMAT,
        "evidence_banks_dir": project_relative(evidence_banks_dir),
        "output_dir": project_relative(output_dir),
        "representation_seed": args.representation_seed,
        "events": len(payloads),
        "total_canonical_evidence_units": sum(
            int(payload["canonical_evidence_unit_count"]) for payload in payloads
        ),
        "warning_count": len(warning_rows),
        "event_ids": sorted(str(payload["event_id"]) for payload in payloads),
    }
    write_json(output_dir / "summary.json", summary)
    if warning_rows:
        (output_dir / "validation_warnings.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
                for row in warning_rows
            ),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
