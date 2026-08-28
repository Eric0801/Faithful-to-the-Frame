#!/usr/bin/env python3
"""Build and audit frozen T6 canonical evidence-order randomizations.

T6 keeps each B0 event's metadata and evidence-unit objects byte-for-byte
identical after canonical JSON serialization.  Its sole intervention is a
pre-generated, event-specific permutation of the top-level evidence-unit list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from render_b0_canonical_evidence_only import (
    CANONICAL_BANK_FIELDS,
    canonical_bank_units,
    event_metadata_lines,
    json_sha256,
    project_relative,
    read_json,
    render_text,
    validate_units,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_BANKS_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "evidence_banks_entity_visible"
)
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "t6_evidence_order_randomization_20260711"
)
DEFAULT_MANIFEST = DEFAULT_ROOT / "t6_order_randomization_manifest.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "t6_payloads_event_level"

TREATMENT_NAME = "T6_canonical_evidence_order_randomized"
UPSTREAM_MODEL_FAMILY = "deterministic_evidence_order_randomization"
RENDERER_VERSION = "t6_canonical_evidence_order_randomization_v1"
PERMUTATION_VERSION = "fisher_yates_nonidentity_v1"
SERIALIZATION_FORMAT = "t2_t3_upstream_canonical_bank_jsonl_no_source_quote"
EXPECTED_MAIN_EVENTS = 94


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-banks-dir", default=str(DEFAULT_EVIDENCE_BANKS_DIR))
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--master-seed", type=int, default=20260711)
    parser.add_argument("--representation-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def unit_identity(unit: dict[str, Any]) -> str:
    evidence_id = unit.get("evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("all evidence units must have a non-empty evidence_id")
    return evidence_id


def unordered_units_sha256(units: list[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(sorted(units, key=unit_identity)))


def derive_seed(master_seed: int, event_id: str) -> tuple[int, str]:
    digest = hashlib.sha256(f"{master_seed}:{event_id}".encode("ascii")).digest()
    return int.from_bytes(digest[:16], "big"), digest.hex()


def nonidentity_permutation(length: int, seed: int) -> list[int]:
    if length < 2:
        raise ValueError("T6 requires at least two evidence units per event")
    rng = random.Random(seed)
    indices = list(range(length))
    for _ in range(100):
        rng.shuffle(indices)
        if any(old_index != new_index for new_index, old_index in enumerate(indices)):
            return indices.copy()
    raise RuntimeError("could not generate a non-identity permutation")


def build_manifest_row(
    bank: dict[str, Any],
    units: list[dict[str, Any]],
    master_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event_id = str(bank["event_id"])
    original_ids = [unit_identity(unit) for unit in units]
    if len(set(original_ids)) != len(original_ids):
        raise ValueError(f"{event_id} has duplicate evidence IDs")
    event_seed, seed_sha256 = derive_seed(master_seed, event_id)
    original_indices = nonidentity_permutation(len(units), event_seed)
    permuted_units = [units[index] for index in original_indices]
    permuted_ids = [unit_identity(unit) for unit in permuted_units]
    if Counter(original_ids) != Counter(permuted_ids):
        raise ValueError(f"{event_id} permutation changed the evidence-ID multiset")
    if original_ids == permuted_ids:
        raise ValueError(f"{event_id} permutation is unexpectedly identity")

    old_to_new = [permuted_ids.index(evidence_id) for evidence_id in original_ids]
    return {
        "event_id": event_id,
        "treatment": TREATMENT_NAME,
        "permutation_version": PERMUTATION_VERSION,
        "master_seed": master_seed,
        "event_seed_sha256": seed_sha256,
        "source_bank_sha256": json_sha256(units),
        "unordered_units_sha256": unordered_units_sha256(units),
        "original_evidence_ids": original_ids,
        "permuted_evidence_ids": permuted_ids,
        "permutation_old_to_new_index": old_to_new,
        "unit_count": len(units),
        "fixed_position_count": sum(
            old_index == new_index for old_index, new_index in enumerate(old_to_new)
        ),
        "original_ordered_units_sha256": json_sha256(units),
        "permuted_ordered_units_sha256": json_sha256(permuted_units),
    }, permuted_units


def validate_permutation(
    event_id: str,
    original_units: list[dict[str, Any]],
    permuted_units: list[dict[str, Any]],
    row: dict[str, Any],
) -> None:
    if len(original_units) != len(permuted_units):
        raise ValueError(f"{event_id} changed evidence-unit count")
    original_by_id = {unit_identity(unit): unit for unit in original_units}
    permuted_by_id = {unit_identity(unit): unit for unit in permuted_units}
    if original_by_id.keys() != permuted_by_id.keys():
        raise ValueError(f"{event_id} changed evidence-unit IDs")
    for evidence_id, original in original_by_id.items():
        if canonical_json_bytes(original) != canonical_json_bytes(permuted_by_id[evidence_id]):
            raise ValueError(f"{event_id}/{evidence_id} changed evidence-unit content")
    if json_sha256(permuted_units) != row["permuted_ordered_units_sha256"]:
        raise ValueError(f"{event_id} permuted-order hash mismatch")
    if unordered_units_sha256(permuted_units) != row["unordered_units_sha256"]:
        raise ValueError(f"{event_id} unordered-unit hash mismatch")


def payload_for_bank(
    bank: dict[str, Any],
    path: Path,
    manifest_row: dict[str, Any],
    permuted_units: list[dict[str, Any]],
    representation_seed: int,
) -> dict[str, Any]:
    event_id = str(bank["event_id"])
    original_units = canonical_bank_units(bank)
    validate_permutation(event_id, original_units, permuted_units, manifest_row)
    original_text = render_text(bank, original_units)
    rendered_text = render_text(bank, permuted_units)
    metadata_prefix = "Event metadata:\n" + "\n".join(event_metadata_lines(bank)) + "\n\nCanonical evidence-unit bank:\n"
    if not original_text.startswith(metadata_prefix) or not rendered_text.startswith(metadata_prefix):
        raise ValueError(f"{event_id} changed metadata or serialization shell")
    return {
        "event_id": event_id,
        "treatment": TREATMENT_NAME,
        "upstream_model_family": UPSTREAM_MODEL_FAMILY,
        "representation_seed": representation_seed,
        "renderer_version": RENDERER_VERSION,
        "serialization_format": SERIALIZATION_FORMAT,
        "permutation_version": PERMUTATION_VERSION,
        "manifest_event_seed_sha256": manifest_row["event_seed_sha256"],
        "rendered_text": rendered_text,
        "evidence_bank_path": project_relative(path),
        "source_packet_path": bank.get("source_packet_path"),
        "canonical_evidence_unit_count": len(permuted_units),
        "canonical_bank_units_sha256": manifest_row["permuted_ordered_units_sha256"],
        "unordered_canonical_bank_units_sha256": manifest_row["unordered_units_sha256"],
        "rendered_text_sha256": sha256_bytes(rendered_text.encode("utf-8")),
        "b0_rendered_text_sha256": sha256_bytes(original_text.encode("utf-8")),
        "b0_ordered_units_sha256": manifest_row["original_ordered_units_sha256"],
        "validation_warnings": validate_units(permuted_units),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def main() -> int:
    args = parse_args()
    banks_dir = Path(args.evidence_banks_dir)
    root = Path(args.root)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    if root.exists() and not args.overwrite:
        raise FileExistsError(f"{root} exists; pass --overwrite")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    manifest_rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for bank_path in sorted(banks_dir.glob("evt_*.json")):
        bank = read_json(bank_path)
        units = canonical_bank_units(bank)
        row, permuted_units = build_manifest_row(bank, units, args.master_seed)
        payload = payload_for_bank(
            bank, bank_path, row, permuted_units, args.representation_seed
        )
        manifest_rows.append(row)
        payloads.append(payload)
        write_json(output_dir / f"{payload['event_id']}_rs{args.representation_seed}.json", payload)
        audit_rows.append(
            {
                "event_id": payload["event_id"],
                "unit_count": row["unit_count"],
                "fixed_position_count": row["fixed_position_count"],
                "content_parity": True,
                "metadata_parity": True,
                "order_changed": True,
                "original_ordered_units_sha256": row["original_ordered_units_sha256"],
                "permuted_ordered_units_sha256": row["permuted_ordered_units_sha256"],
                "unordered_units_sha256": row["unordered_units_sha256"],
            }
        )

    if len(payloads) != EXPECTED_MAIN_EVENTS:
        raise ValueError(f"expected {EXPECTED_MAIN_EVENTS} events; found {len(payloads)}")
    write_jsonl(manifest_path, manifest_rows)
    write_jsonl(root / "t6_order_randomization_audit.jsonl", audit_rows)
    summary = {
        "treatment": TREATMENT_NAME,
        "upstream_model_family": UPSTREAM_MODEL_FAMILY,
        "renderer_version": RENDERER_VERSION,
        "permutation_version": PERMUTATION_VERSION,
        "master_seed": args.master_seed,
        "events": len(payloads),
        "total_canonical_evidence_units": sum(
            int(payload["canonical_evidence_unit_count"]) for payload in payloads
        ),
        "all_events_changed_order": all(
            row["original_evidence_ids"] != row["permuted_evidence_ids"]
            for row in manifest_rows
        ),
        "content_parity_passed": True,
        "metadata_parity_passed": True,
        "manifest_path": project_relative(manifest_path),
        "payload_dir": project_relative(output_dir),
    }
    write_json(root / "t6_order_randomization_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
