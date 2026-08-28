#!/usr/bin/env python3
"""Render deterministic T5 linguistic-deframing payloads from B0 evidence.

The reviewed JSONL manifest must contain exactly one object for every
``event_id`` / ``evidence_id`` in the selected canonical evidence banks. Each
object must include ``event_id``, ``evidence_id``, ``original_claim`` (or the
legacy alias ``b0_claim``), ``category``, ``value``, ``source_ids``,
``support_label``, ``approved``, and ``proposed_t5_claim``. Manifest fields
other than ``proposed_t5_claim`` must exactly match the canonical bank.

For approved records, ``proposed_t5_claim`` replaces the original claim. For
all other records, the original canonical claim is rendered unchanged. T5 is
therefore a paired linguistic intervention over the B0 representation rather
than a new evidence selection or presentation format.
"""

from __future__ import annotations

import argparse
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
    / "t5_linguistic_deframing_20260710"
    / "t5_payloads_event_level"
)

TREATMENT_NAME = "T5_linguistic_deframing"
UPSTREAM_MODEL_FAMILY = "deterministic_linguistic_deframing"
RENDERER_VERSION = "t5_linguistic_deframing_v1_b0_matched_jsonl"
SERIALIZATION_FORMAT = "t2_t3_upstream_canonical_bank_jsonl_no_source_quote"
CANONICAL_BANK_FIELDS = (
    "evidence_id",
    "category",
    "claim",
    "value",
    "source_ids",
    "support_label",
)
MANIFEST_MATCH_FIELDS = (
    "category",
    "value",
    "source_ids",
    "support_label",
)
REQUIRED_MANIFEST_FIELDS = (
    "event_id",
    "evidence_id",
    "category",
    "value",
    "source_ids",
    "support_label",
    "approved",
    "proposed_t5_claim",
)
NUMERIC_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[+-]?\$?\d[\d,]*(?:\.\d+)?%?|\d+(?:st|nd|rd|th))(?![A-Za-z0-9_])"
)
DATE_TOKEN_RE = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:,\s*\d{4})?|"
    r"(?:Q[1-4]|FY)\s*\d{2,4})\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-banks-dir", default=str(DEFAULT_EVIDENCE_BANKS_DIR))
    parser.add_argument(
        "--transformation-manifest",
        required=True,
        help="Reviewed JSONL with one record per event_id/evidence_id.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--representation-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


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
    return {field: unit.get(field) for field in CANONICAL_BANK_FIELDS}


def canonical_bank_units(bank: dict[str, Any]) -> list[dict[str, Any]]:
    units = bank.get("evidence_units")
    if not isinstance(units, list):
        raise ValueError(f"{bank.get('event_id', '<unknown>')} evidence_units must be a list")
    canonical_units: list[dict[str, Any]] = []
    for index, unit in enumerate(units, start=1):
        if not isinstance(unit, dict):
            raise ValueError(f"{bank.get('event_id', '<unknown>')} unit {index} is not an object")
        missing = [field for field in CANONICAL_BANK_FIELDS if field not in unit]
        if missing:
            raise ValueError(
                f"{bank.get('event_id', '<unknown>')} unit {index} missing fields: {', '.join(missing)}"
            )
        canonical_units.append(canonical_bank_unit(unit))
    return canonical_units


def serialize_evidence_bank_units(units: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(unit, ensure_ascii=True, sort_keys=True) for unit in units)


def render_text(bank: dict[str, Any], units: list[dict[str, Any]]) -> str:
    return (
        "Event metadata:\n"
        + "\n".join(event_metadata_lines(bank))
        + "\n\n"
        + "Canonical evidence-unit bank:\n"
        + serialize_evidence_bank_units(units)
    )


def manifest_original_claim(record: dict[str, Any], line_number: int) -> str:
    original_claim = record.get("original_claim")
    legacy_claim = record.get("b0_claim")
    if original_claim is None and legacy_claim is None:
        raise ValueError(f"manifest line {line_number} missing original_claim")
    if original_claim is not None and legacy_claim is not None and original_claim != legacy_claim:
        raise ValueError(f"manifest line {line_number} original_claim and b0_claim disagree")
    claim = original_claim if original_claim is not None else legacy_claim
    if not isinstance(claim, str):
        raise ValueError(f"manifest line {line_number} original claim must be a string")
    return claim


def manifest_key(record: dict[str, Any], line_number: int) -> tuple[str, str]:
    event_id = record.get("event_id")
    evidence_id = record.get("evidence_id")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError(f"manifest line {line_number} has invalid event_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError(f"manifest line {line_number} has invalid evidence_id")
    return event_id, evidence_id


def read_manifest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"transformation manifest not found: {path}")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"manifest line {line_number} is blank")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"manifest line {line_number} is not valid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"manifest line {line_number} must be a JSON object")
        missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in record]
        if missing:
            raise ValueError(f"manifest line {line_number} missing fields: {', '.join(missing)}")
        key = manifest_key(record, line_number)
        manifest_original_claim(record, line_number)
        if not isinstance(record["approved"], bool):
            raise ValueError(f"manifest line {line_number} approved must be boolean")
        if not isinstance(record["proposed_t5_claim"], str):
            raise ValueError(f"manifest line {line_number} proposed_t5_claim must be a string")
        if key in records:
            raise ValueError(f"duplicate manifest key: {key[0]}/{key[1]}")
        records[key] = record
    if not records:
        raise ValueError(f"transformation manifest is empty: {path}")
    return records


def validate_bank_units(event_id: str, units: list[dict[str, Any]]) -> None:
    seen_evidence_ids: set[str] = set()
    for index, unit in enumerate(units, start=1):
        missing = [field for field in CANONICAL_BANK_FIELDS if field not in unit]
        if missing:
            raise ValueError(f"{event_id} unit {index} missing fields: {', '.join(missing)}")
        evidence_id = unit["evidence_id"]
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError(f"{event_id} unit {index} has invalid evidence_id")
        if evidence_id in seen_evidence_ids:
            raise ValueError(f"{event_id} has duplicate evidence_id: {evidence_id}")
        seen_evidence_ids.add(evidence_id)
        if not isinstance(unit["claim"], str):
            raise ValueError(f"{event_id}/{evidence_id} claim must be a string")
        if not isinstance(unit["source_ids"], list) or not unit["source_ids"]:
            raise ValueError(f"{event_id}/{evidence_id} source_ids must be a non-empty list")


def protected_claim_tokens(claim: str, bank: dict[str, Any], unit: dict[str, Any]) -> Counter[str]:
    tokens = [match.group(0).casefold() for match in NUMERIC_TOKEN_RE.finditer(claim)]
    tokens.extend(match.group(0).casefold() for match in DATE_TOKEN_RE.finditer(claim))
    identifiers = [
        str(bank.get("event_id", "")),
        str(bank.get("ticker", "")),
        str(unit.get("evidence_id", "")),
        *(str(source_id) for source_id in unit.get("source_ids", [])),
    ]
    claim_folded = claim.casefold()
    for identifier in identifiers:
        normalized = identifier.strip()
        if normalized and normalized.casefold() in claim_folded:
            tokens.append(normalized.casefold())
    return Counter(tokens)


def validate_protected_claim_content(
    original_claim: str,
    proposed_claim: str,
    bank: dict[str, Any],
    unit: dict[str, Any],
) -> None:
    original_tokens = protected_claim_tokens(original_claim, bank, unit)
    proposed_tokens = protected_claim_tokens(proposed_claim, bank, unit)
    missing_tokens = original_tokens - proposed_tokens
    added_tokens = proposed_tokens - original_tokens
    if missing_tokens or added_tokens:
        raise ValueError(
            f"{bank['event_id']}/{unit['evidence_id']} alters protected claim tokens: "
            f"missing={dict(missing_tokens)}, added={dict(added_tokens)}"
        )


def apply_manifest_record(
    bank: dict[str, Any],
    unit: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    event_id = str(bank["event_id"])
    evidence_id = str(unit["evidence_id"])
    if manifest_original_claim(record, 0) != unit["claim"]:
        raise ValueError(f"{event_id}/{evidence_id} manifest original claim does not exactly match bank")
    for field in MANIFEST_MATCH_FIELDS:
        if record[field] != unit[field]:
            raise ValueError(f"{event_id}/{evidence_id} manifest {field} does not exactly match bank")
    if not record["approved"]:
        return dict(unit)
    proposed_claim = record["proposed_t5_claim"]
    validate_protected_claim_content(unit["claim"], proposed_claim, bank, unit)
    transformed = dict(unit)
    transformed["claim"] = proposed_claim
    return transformed


def whitespace_token_count(value: str) -> int:
    return len(value.split())


def payload_for_bank(
    bank: dict[str, Any],
    evidence_bank_path: Path,
    manifest: dict[tuple[str, str], dict[str, Any]],
    representation_seed: int,
) -> dict[str, Any]:
    event_id = bank.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError(f"{evidence_bank_path} has invalid event_id")
    units = canonical_bank_units(bank)
    validate_bank_units(event_id, units)
    transformed_units: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    for unit in units:
        key = (event_id, unit["evidence_id"])
        record = manifest.get(key)
        if record is None:
            raise ValueError(f"missing manifest key: {event_id}/{unit['evidence_id']}")
        transformed_unit = apply_manifest_record(bank, unit, record)
        transformed_units.append(transformed_unit)
        if transformed_unit["claim"] != unit["claim"]:
            changed_rows.append(
                {
                    "event_id": event_id,
                    "evidence_id": unit["evidence_id"],
                    "original_claim": unit["claim"],
                    "proposed_t5_claim": transformed_unit["claim"],
                    "char_delta": len(transformed_unit["claim"]) - len(unit["claim"]),
                    "token_delta": whitespace_token_count(transformed_unit["claim"])
                    - whitespace_token_count(unit["claim"]),
                }
            )
    original_text = render_text(bank, units)
    rendered_text = render_text(bank, transformed_units)
    return {
        "event_id": event_id,
        "treatment": TREATMENT_NAME,
        "upstream_model_family": UPSTREAM_MODEL_FAMILY,
        "representation_seed": representation_seed,
        "renderer_version": RENDERER_VERSION,
        "serialization_format": SERIALIZATION_FORMAT,
        "rendered_text": rendered_text,
        "evidence_bank_path": project_relative(evidence_bank_path),
        "source_packet_path": bank.get("source_packet_path"),
        "canonical_evidence_unit_count": len(transformed_units),
        "canonical_bank_units_sha256": json_sha256(transformed_units),
        "rendered_text_sha256": hashlib.sha256(rendered_text.encode("utf-8")).hexdigest(),
        "b0_rendered_text_sha256": hashlib.sha256(original_text.encode("utf-8")).hexdigest(),
        "changed_row_count": len(changed_rows),
        "claim_char_delta": sum(row["char_delta"] for row in changed_rows),
        "claim_token_delta": sum(row["token_delta"] for row in changed_rows),
        "rendered_text_char_delta": len(rendered_text) - len(original_text),
        "rendered_text_token_delta": whitespace_token_count(rendered_text)
        - whitespace_token_count(original_text),
        "changed_rows": changed_rows,
        "validation_warnings": [],
    }


def validate_manifest_coverage(
    manifest: dict[tuple[str, str], dict[str, Any]],
    bank_paths: list[Path],
) -> None:
    expected_keys: set[tuple[str, str]] = set()
    for bank_path in bank_paths:
        bank = read_json(bank_path)
        event_id = bank.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError(f"{bank_path} has invalid event_id")
        units = canonical_bank_units(bank)
        validate_bank_units(event_id, units)
        expected_keys.update((event_id, str(unit["evidence_id"])) for unit in units)
    manifest_keys = set(manifest)
    missing = sorted(expected_keys - manifest_keys)
    extras = sorted(manifest_keys - expected_keys)
    if missing or extras:
        pieces: list[str] = []
        if missing:
            pieces.append("missing=" + ", ".join(f"{event}/{evidence}" for event, evidence in missing[:10]))
        if extras:
            pieces.append("extra=" + ", ".join(f"{event}/{evidence}" for event, evidence in extras[:10]))
        raise ValueError("manifest coverage mismatch: " + "; ".join(pieces))


def main() -> int:
    args = parse_args()
    evidence_banks_dir = Path(args.evidence_banks_dir)
    manifest_path = Path(args.transformation_manifest)
    output_dir = Path(args.output_dir)
    bank_paths = sorted(evidence_banks_dir.glob("evt_*.json"))
    if not bank_paths:
        raise ValueError(f"no evt_*.json evidence banks found in {evidence_banks_dir}")
    manifest = read_manifest(manifest_path)
    validate_manifest_coverage(manifest, bank_paths)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    for bank_path in bank_paths:
        payload = payload_for_bank(bank=read_json(bank_path), evidence_bank_path=bank_path, manifest=manifest, representation_seed=args.representation_seed)
        payload_path = output_dir / f"{payload['event_id']}_rs{args.representation_seed}.json"
        write_json(payload_path, payload)
        payloads.append(payload)
        changed_rows.extend(payload["changed_rows"])

    summary = {
        "treatment": TREATMENT_NAME,
        "upstream_model_family": UPSTREAM_MODEL_FAMILY,
        "renderer_version": RENDERER_VERSION,
        "serialization_format": SERIALIZATION_FORMAT,
        "evidence_banks_dir": project_relative(evidence_banks_dir),
        "transformation_manifest": project_relative(manifest_path),
        "transformation_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "output_dir": project_relative(output_dir),
        "representation_seed": args.representation_seed,
        "events": len(payloads),
        "total_canonical_evidence_units": sum(int(payload["canonical_evidence_unit_count"]) for payload in payloads),
        "changed_row_count": len(changed_rows),
        "claim_char_delta": sum(int(payload["claim_char_delta"]) for payload in payloads),
        "claim_token_delta": sum(int(payload["claim_token_delta"]) for payload in payloads),
        "rendered_text_char_delta": sum(int(payload["rendered_text_char_delta"]) for payload in payloads),
        "rendered_text_token_delta": sum(int(payload["rendered_text_token_delta"]) for payload in payloads),
        "event_ids": sorted(str(payload["event_id"]) for payload in payloads),
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "changed_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in changed_rows),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
