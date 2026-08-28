#!/usr/bin/env python3
"""Validate recovered E1 decision rows against their rendered treatment packets.

E1 has three prior-only arms, where citations must be empty, and one evidence
replay arm, where citations must name IDs that are visible in the exact replay
packet.  In addition to ordinary ``S###`` source-unit IDs, the replay packet
can expose ``X###`` structured-fact IDs.  This validator intentionally checks
the rendered packet rather than an upstream evidence-bank schema so that the
validation target is exactly what each receiver saw.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


E1_ARMS = (
    "E1_prior_visible",
    "E1_prior_issuer_masked",
    "E1_prior_context_minimal",
    "E1_t1_evidence_visible_replay",
)
REPLAY_ARM = "E1_t1_evidence_visible_replay"
PRIOR_ONLY_ARMS = frozenset(E1_ARMS) - {REPLAY_ARM}
VISIBLE_ID_PATTERN = re.compile(r"(?m)^-\s+([SX]\d{3})\b")
REQUIRED_FIELDS = (
    "event_id",
    "treatment",
    "upstream_model_family",
    "profile",
    "profile_group",
    "model_family",
    "representation_seed",
    "decision_seed",
    "evidence_used",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="Materialized E1 decision-row CSV.")
    parser.add_argument(
        "--rendered-packets",
        required=True,
        help="Directory containing E1 arm directories and rendered event JSON files.",
    )
    parser.add_argument("--report", required=True, help="Destination for the JSON validation report.")
    parser.add_argument("--expected-total", type=int, default=0)
    parser.add_argument("--expected-events", type=int, default=0)
    parser.add_argument("--expected-profiles", type=int, default=0)
    parser.add_argument(
        "--expected-model-families",
        default="",
        help="Comma-separated required model-family names. Omit to infer them from input.",
    )
    return parser.parse_args()


def parse_json_list(value: str, *, row_number: int) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"row {row_number}: evidence_used is not valid JSON") from error
    if not isinstance(parsed, list) or not all(isinstance(item, str) and item.strip() for item in parsed):
        raise ValueError(f"row {row_number}: evidence_used must be a JSON list of non-empty strings")
    return [item.strip() for item in parsed]


def visible_ids_for_event(rendered_root: Path, event_id: str) -> set[str]:
    packet_path = rendered_root / REPLAY_ARM / f"{event_id}.json"
    try:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{packet_path}: unable to read rendered replay packet: {error}") from error
    if packet.get("event_id") != event_id or packet.get("treatment") != REPLAY_ARM:
        raise ValueError(f"{packet_path}: does not match E1 replay event {event_id}")
    rendered_text = packet.get("rendered_text")
    if not isinstance(rendered_text, str):
        raise ValueError(f"{packet_path}: rendered_text is missing")
    return set(VISIBLE_ID_PATTERN.findall(rendered_text))


def error(row_number: int, code: str, message: str) -> dict[str, Any]:
    return {"row_number": row_number, "code": code, "message": message}


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv)
    rendered_root = Path(args.rendered_packets)
    expected_models = {item.strip() for item in args.expected_model_families.split(",") if item.strip()}
    errors: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    visible_ids: dict[str, set[str]] = {}

    with input_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path}: missing header row")
        missing = sorted(set(REQUIRED_FIELDS) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{input_path}: missing required columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            rows.append(row)
            missing_values = [field for field in REQUIRED_FIELDS if not str(row.get(field, "")).strip()]
            if missing_values:
                errors.append(error(row_number, "missing_required_value", ", ".join(missing_values)))
                continue
            treatment = row["treatment"].strip()
            if treatment not in E1_ARMS:
                errors.append(error(row_number, "unknown_treatment", treatment))
                continue
            if row["upstream_model_family"].strip() != "e1_prior_hierarchy":
                errors.append(error(row_number, "unexpected_upstream_model_family", row["upstream_model_family"]))
            try:
                cited_ids = parse_json_list(row["evidence_used"], row_number=row_number)
            except ValueError as exc:
                errors.append(error(row_number, "invalid_evidence_used", str(exc)))
                continue
            if treatment in PRIOR_ONLY_ARMS and cited_ids:
                errors.append(error(row_number, "prior_arm_has_citations", ", ".join(cited_ids)))
            if treatment == REPLAY_ARM:
                if not cited_ids:
                    errors.append(error(row_number, "replay_arm_missing_citations", "evidence_used is empty"))
                    continue
                event_id = row["event_id"].strip()
                if event_id not in visible_ids:
                    try:
                        visible_ids[event_id] = visible_ids_for_event(rendered_root, event_id)
                    except ValueError as exc:
                        errors.append(error(row_number, "rendered_packet_error", str(exc)))
                        continue
                unavailable = sorted(set(cited_ids) - visible_ids[event_id])
                if unavailable:
                    errors.append(
                        error(
                            row_number,
                            "citation_not_visible_in_replay_packet",
                            ", ".join(unavailable),
                        )
                    )

    row_keys = Counter(
        (
            row.get("event_id", "").strip(),
            row.get("treatment", "").strip(),
            row.get("profile", "").strip(),
            row.get("model_family", "").strip(),
            row.get("representation_seed", "").strip(),
            row.get("decision_seed", "").strip(),
        )
        for row in rows
    )
    duplicate_keys = [key for key, count in row_keys.items() if count > 1]
    if duplicate_keys:
        errors.append(error(0, "duplicate_decision_identity", f"{len(duplicate_keys)} duplicate identities"))

    events = {row.get("event_id", "").strip() for row in rows if row.get("event_id", "").strip()}
    models = {row.get("model_family", "").strip() for row in rows if row.get("model_family", "").strip()}
    profiles_by_event_model: dict[tuple[str, str], set[str]] = defaultdict(set)
    arms_by_event_model_profile: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        event_id = row.get("event_id", "").strip()
        model = row.get("model_family", "").strip()
        profile = row.get("profile", "").strip()
        treatment = row.get("treatment", "").strip()
        if event_id and model and profile:
            profiles_by_event_model[(event_id, model)].add(profile)
            arms_by_event_model_profile[(event_id, model, profile)].add(treatment)
    incomplete_cells = [
        key for key, observed in arms_by_event_model_profile.items() if observed != set(E1_ARMS)
    ]
    if incomplete_cells:
        errors.append(error(0, "incomplete_matched_cells", f"{len(incomplete_cells)} cells missing E1 arms"))
    if args.expected_profiles and any(len(profiles) != args.expected_profiles for profiles in profiles_by_event_model.values()):
        errors.append(error(0, "unexpected_profile_count", f"expected {args.expected_profiles} profiles per event/model"))
    if args.expected_events and len(events) != args.expected_events:
        errors.append(error(0, "unexpected_event_count", f"expected {args.expected_events}, found {len(events)}"))
    if expected_models and models != expected_models:
        errors.append(error(0, "unexpected_model_families", f"expected {sorted(expected_models)}, found {sorted(models)}"))
    if args.expected_total and len(rows) != args.expected_total:
        errors.append(error(0, "unexpected_row_count", f"expected {args.expected_total}, found {len(rows)}"))

    invalid_row_numbers = {item["row_number"] for item in errors if item["row_number"] > 0}
    report = {
        "input_csv": str(input_path),
        "rendered_packets": str(rendered_root),
        "summary": {
            "all_valid": not errors,
            "total_rows": len(rows),
            "valid_rows": len(rows) - len(invalid_row_numbers),
            "invalid_rows": len(invalid_row_numbers),
            "events": len(events),
            "model_families": sorted(models),
            "treatment_counts": dict(sorted(Counter(row.get("treatment", "") for row in rows).items())),
            "source_packet_validation_enabled": True,
        },
        "errors": errors,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
