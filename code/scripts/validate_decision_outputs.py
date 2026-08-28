#!/usr/bin/env python3
"""Validate downstream decision outputs against the implementation spec."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "decisions.jsonl"

REQUIRED_FIELDS = (
    "event_id",
    "treatment",
    "profile",
    "profile_group",
    "model_family",
    "representation_seed",
    "decision_seed",
    "expected_return_5d",
    "confidence",
    "action",
    "action_strength",
    "key_reasons",
    "evidence_used",
    "uncertainty_notes",
)

TREATMENT_FAMILIES = {
    "T1_full": "T1",
    "T1_length_matched": "T1",
    "T1_raw_public_information": "T1",
    "T2_shared_summary": "T2",
    "T2_shared_llm_summary": "T2",
    "T3_independent": "T3",
    "T3_independent_summary": "T3",
    "T3_independent_summaries": "T3",
    "T3_independent_llm_summary": "T3",
    "T4_structured": "T4",
    "T4_structured_evidence_card": "T4",
    "T4_shared_atomic_evidence_view_control": "T4",
    "T4_full_structured_evidence_ledger": "T4",
    "T4_SAEV_deterministic": "T4",
    "B0_canonical_evidence_only": "B0",
    "T5_linguistic_deframing": "T5",
    "T6_canonical_evidence_order_randomized": "T6",
}

PROFILE_GROUP_BY_PROFILE = {
    "day_trader": "retail",
    "retail_day_trader": "retail",
    "swing_trader": "retail",
    "retail_swing_trader": "retail",
    "long_term_retail": "retail",
    "retail_long_term_fundamental": "retail",
    "event_driven_hedge_fund": "institutional",
    "institutional_event_driven_hedge_fund": "institutional",
    "prop_trading": "institutional",
    "institutional_prop_trader": "institutional",
    "institutional_prop_trading": "institutional",
    "investment_advisor": "institutional",
    "institutional_investment_advisor": "institutional",
    "unprofiled_baseline": "unprofiled",
}

ALLOWED_PROFILE_GROUPS = ("retail", "institutional", "unprofiled")
ALLOWED_MODEL_FAMILIES = (
    "claude-sonnet-4.5",
    "gpt-5.2",
    "qwen3-235b-a22b",
    "deepseek-v3.1",
)
ALLOWED_UPSTREAM_MODEL_FAMILIES = (
    "none",
    "deterministic_canonical_evidence",
    "deterministic_linguistic_deframing",
    "deterministic_evidence_order_randomization",
    "deterministic_full_ledger",
    "claude-sonnet-4.5",
    "gpt-5.2",
    "qwen3-235b-a22b",
    "deepseek-v3.1",
    "legacy_unspecified",
    "mock",
)
ALLOWED_ACTIONS = ("buy", "hold", "sell")
ALLOWED_DECISION_SEEDS = {1, 2}
ALLOWED_DETERMINISTIC_REPRESENTATION_SEEDS = {0}
ALLOWED_NON_T1_REPRESENTATION_SEEDS = {0, 1, 2}
SOURCE_PACKET_SUFFIXES = {".json", ".jsonl"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT))
    parser.add_argument(
        "--report",
        default=None,
        help="Path to the JSON validation report. Defaults next to the input file.",
    )
    parser.add_argument(
        "--source-packets",
        default=None,
        help="Optional source-packet JSON/JSONL file or directory for evidence_used checks.",
    )
    return parser.parse_args()


def default_report_path(input_path: Path) -> Path:
    if input_path.suffix:
        return input_path.with_name(f"{input_path.stem}.validation_report.json")
    return input_path.with_name(f"{input_path.name}.validation_report.json")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def make_error(
    field: str,
    code: str,
    message: str,
    value: Any | None = None,
) -> dict[str, Any]:
    error = {"field": field, "code": code, "message": message}
    if value is not None:
        error["received"] = summarize_value(value)
    return error


def summarize_value(value: Any, max_length: int = 160) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_length else value[: max_length - 3] + "..."
    if isinstance(value, list):
        preview = value[:5]
        suffix = [] if len(value) <= 5 else [f"... ({len(value)} items total)"]
        return preview + suffix
    if isinstance(value, dict):
        items = list(value.items())[:5]
        preview = {key: val for key, val in items}
        if len(value) > 5:
            preview["..."] = f"{len(value)} keys total"
        return preview
    return repr(value)


def load_input_records(path: Path) -> tuple[list[tuple[int, Any]], list[dict[str, Any]], str]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return [], [], "empty"
    if stripped[0] == "[":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("top-level JSON payload must be an array")
        return [(idx + 1, item) for idx, item in enumerate(payload)], [], "json_array"
    if path.suffix.lower() == ".jsonl":
        rows, errors = load_jsonl_records(text)
        return rows, errors, "jsonl"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows, errors = load_jsonl_records(text)
        return rows, errors, "jsonl"
    if isinstance(payload, dict):
        return [(1, payload)], [], "json_object"
    if isinstance(payload, list):
        return [(idx + 1, item) for idx, item in enumerate(payload)], [], "json_array"
    raise ValueError("input must be JSONL, a JSON array, or a single JSON object")


def load_jsonl_records(text: str) -> tuple[list[tuple[int, Any]], list[dict[str, Any]]]:
    rows: list[tuple[int, Any]] = []
    errors: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            rows.append((line_number, json.loads(stripped)))
        except json.JSONDecodeError as exc:
            errors.append(
                {
                    "line_number": line_number,
                    "code": "invalid_jsonl_line",
                    "message": f"{exc.msg} at column {exc.colno}",
                    "raw_line": summarize_value(stripped),
                }
            )
    return rows, errors


def iter_source_packet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SOURCE_PACKET_SUFFIXES
    )


def extract_source_ids(record: dict[str, Any]) -> set[str]:
    source_ids: set[str] = set()
    for field in ("source_units", "xbrl_facts"):
        items = record.get(field)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            source_id = item.get("source_id")
            if is_nonempty_string(source_id):
                source_ids.add(source_id.strip())
    return source_ids


def load_source_packet_index(
    path: Path,
) -> tuple[dict[str, set[str]], list[dict[str, Any]], int]:
    packet_index: dict[str, set[str]] = {}
    warnings: list[dict[str, Any]] = []
    files = iter_source_packet_files(path)
    for packet_file in files:
        try:
            rows, errors, _ = load_input_records(packet_file)
        except Exception as exc:  # pragma: no cover - defensive reporting path
            warnings.append(
                {
                    "path": str(packet_file),
                    "code": "source_packet_file_unreadable",
                    "message": str(exc),
                }
            )
            continue
        for error in errors:
            warnings.append(
                {
                    "path": str(packet_file),
                    "code": error["code"],
                    "message": error["message"],
                    "line_number": error["line_number"],
                }
            )
        for row_number, record in rows:
            if not isinstance(record, dict):
                continue
            if "source_units" not in record and "xbrl_facts" not in record:
                continue
            event_id = record.get("event_id")
            if not is_nonempty_string(event_id):
                warnings.append(
                    {
                        "path": str(packet_file),
                        "row_number": row_number,
                        "code": "source_packet_missing_event_id",
                        "message": "source-packet record is missing a usable event_id",
                    }
                )
                continue
            source_ids = extract_source_ids(record)
            if not source_ids:
                warnings.append(
                    {
                        "path": str(packet_file),
                        "row_number": row_number,
                        "event_id": event_id,
                        "code": "source_packet_missing_source_ids",
                        "message": "source-packet record did not expose any source_id values",
                    }
                )
                continue
            packet_index.setdefault(event_id.strip(), set()).update(source_ids)
    return packet_index, warnings, len(files)


def validate_string_enum(
    errors: list[dict[str, Any]],
    field: str,
    value: Any,
    allowed_values: tuple[str, ...],
) -> str | None:
    if not is_nonempty_string(value):
        errors.append(
            make_error(field, "invalid_type", "must be a non-empty string", value)
        )
        return None
    text = value.strip()
    if text not in allowed_values:
        errors.append(
            make_error(
                field,
                "invalid_enum",
                f"must be one of {', '.join(allowed_values)}",
                value,
            )
        )
        return None
    return text


def validate_number_range(
    errors: list[dict[str, Any]],
    field: str,
    value: Any,
    minimum: float,
    maximum: float,
) -> float | None:
    if not is_number(value):
        errors.append(make_error(field, "invalid_type", "must be a finite number", value))
        return None
    numeric = float(value)
    if numeric < minimum or numeric > maximum:
        errors.append(
            make_error(
                field,
                "out_of_range",
                f"must be between {minimum} and {maximum}",
                value,
            )
        )
        return None
    return numeric


def validate_integer_enum(
    errors: list[dict[str, Any]],
    field: str,
    value: Any,
    allowed_values: set[int],
) -> int | None:
    if not is_integer(value):
        errors.append(make_error(field, "invalid_type", "must be an integer", value))
        return None
    if value not in allowed_values:
        allowed_text = ", ".join(str(item) for item in sorted(allowed_values))
        errors.append(
            make_error(
                field,
                "invalid_enum",
                f"must be one of {allowed_text}",
                value,
            )
        )
        return None
    return value


def validate_string_list(
    errors: list[dict[str, Any]],
    field: str,
    value: Any,
    minimum_items: int | None = None,
    maximum_items: int | None = None,
) -> list[str] | None:
    if not isinstance(value, list):
        errors.append(make_error(field, "invalid_type", "must be a list", value))
        return None
    items: list[str] = []
    for idx, item in enumerate(value):
        if not is_nonempty_string(item):
            errors.append(
                make_error(
                    field,
                    "invalid_item",
                    f"item {idx} must be a non-empty string",
                    item,
                )
            )
            continue
        items.append(item.strip())
    if minimum_items is not None and len(items) < minimum_items:
        errors.append(
            make_error(
                field,
                "too_few_items",
                f"must contain at least {minimum_items} items",
                value,
            )
        )
    if maximum_items is not None and len(items) > maximum_items:
        errors.append(
            make_error(
                field,
                "too_many_items",
                f"must contain at most {maximum_items} items",
                value,
            )
        )
    return items


def request_upstream_model_family(record: dict[str, Any]) -> Any | None:
    request_metadata = record.get("request_metadata")
    if not isinstance(request_metadata, dict):
        return None
    passthrough_fields = request_metadata.get("passthrough_fields")
    if not isinstance(passthrough_fields, dict):
        return None
    return passthrough_fields.get("upstream_model_family")


def validate_record(
    row_number: int,
    record: Any,
    source_packet_index: dict[str, set[str]] | None,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {
            "row_number": row_number,
            "event_id": None,
            "errors": [
                make_error(
                    "row",
                    "invalid_type",
                    "each decision record must be a JSON object",
                    record,
                )
            ],
        }

    errors: list[dict[str, Any]] = []
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(make_error(field, "missing_field", "field is required"))

    event_id: str | None = None
    if "event_id" in record:
        if not is_nonempty_string(record["event_id"]):
            errors.append(
                make_error(
                    "event_id",
                    "invalid_type",
                    "must be a non-empty string",
                    record["event_id"],
                )
            )
        else:
            event_id = record["event_id"].strip()

    treatment: str | None = None
    treatment_family: str | None = None
    if "treatment" in record:
        treatment = validate_string_enum(
            errors,
            "treatment",
            record["treatment"],
            tuple(sorted(TREATMENT_FAMILIES)),
        )
        if treatment is not None:
            treatment_family = TREATMENT_FAMILIES[treatment]

    profile: str | None = None
    expected_profile_group: str | None = None
    if "profile" in record:
        profile = validate_string_enum(
            errors,
            "profile",
            record["profile"],
            tuple(sorted(PROFILE_GROUP_BY_PROFILE)),
        )
        if profile is not None:
            expected_profile_group = PROFILE_GROUP_BY_PROFILE[profile]

    profile_group: str | None = None
    if "profile_group" in record:
        profile_group = validate_string_enum(
            errors,
            "profile_group",
            record["profile_group"],
            ALLOWED_PROFILE_GROUPS,
        )
        if (
            profile_group is not None
            and expected_profile_group is not None
            and profile_group != expected_profile_group
        ):
            errors.append(
                make_error(
                    "profile_group",
                    "profile_group_mismatch",
                    f"must be {expected_profile_group} for profile {profile}",
                    record["profile_group"],
                )
            )

    if "model_family" in record:
        validate_string_enum(
            errors,
            "model_family",
            record["model_family"],
            ALLOWED_MODEL_FAMILIES,
        )

    request_upstream_family = request_upstream_model_family(record)
    upstream_family: str | None = None
    if "upstream_model_family" in record:
        upstream_family = validate_string_enum(
            errors,
            "upstream_model_family",
            record["upstream_model_family"],
            ALLOWED_UPSTREAM_MODEL_FAMILIES,
        )
    elif request_upstream_family is not None:
        errors.append(
            make_error(
                "upstream_model_family",
                "missing_field",
                "field is required when request metadata includes upstream_model_family",
            )
        )
    if request_upstream_family is not None:
        request_upstream_text = validate_string_enum(
            errors,
            "request_metadata.passthrough_fields.upstream_model_family",
            request_upstream_family,
            ALLOWED_UPSTREAM_MODEL_FAMILIES,
        )
        if (
            upstream_family is not None
            and request_upstream_text is not None
            and upstream_family != request_upstream_text
        ):
            errors.append(
                make_error(
                    "upstream_model_family",
                    "upstream_model_family_mismatch",
                    "must match request_metadata.passthrough_fields.upstream_model_family",
                    record["upstream_model_family"],
                )
            )

    if "representation_seed" in record:
        if treatment_family in {"T1", "B0"}:
            allowed_representation_seeds = ALLOWED_DETERMINISTIC_REPRESENTATION_SEEDS
        elif treatment_family in {"T2", "T3", "T4"}:
            allowed_representation_seeds = ALLOWED_NON_T1_REPRESENTATION_SEEDS
        else:
            allowed_representation_seeds = (
                ALLOWED_DETERMINISTIC_REPRESENTATION_SEEDS
                | ALLOWED_NON_T1_REPRESENTATION_SEEDS
            )
        validate_integer_enum(
            errors,
            "representation_seed",
            record["representation_seed"],
            allowed_representation_seeds,
        )

    if "decision_seed" in record:
        validate_integer_enum(
            errors,
            "decision_seed",
            record["decision_seed"],
            ALLOWED_DECISION_SEEDS,
        )

    if "expected_return_5d" in record:
        validate_number_range(
            errors,
            "expected_return_5d",
            record["expected_return_5d"],
            -0.20,
            0.20,
        )

    if "confidence" in record:
        validate_number_range(errors, "confidence", record["confidence"], 0.0, 1.0)

    if "action" in record:
        validate_string_enum(errors, "action", record["action"], ALLOWED_ACTIONS)

    if "action_strength" in record:
        validate_number_range(
            errors,
            "action_strength",
            record["action_strength"],
            -1.0,
            1.0,
        )

    if "key_reasons" in record:
        validate_string_list(
            errors,
            "key_reasons",
            record["key_reasons"],
            minimum_items=2,
            maximum_items=5,
        )

    evidence_used: list[str] | None = None
    if "evidence_used" in record:
        evidence_used = validate_string_list(
            errors,
            "evidence_used",
            record["evidence_used"],
            minimum_items=1,
        )
        if evidence_used and source_packet_index is not None and event_id is not None:
            valid_source_ids = source_packet_index.get(event_id)
            if valid_source_ids is None:
                errors.append(
                    make_error(
                        "evidence_used",
                        "missing_source_packet",
                        f"no source packet found for event_id {event_id}",
                    )
                )
            else:
                missing_ids = sorted(
                    source_id for source_id in evidence_used if source_id not in valid_source_ids
                )
                if missing_ids:
                    errors.append(
                        make_error(
                            "evidence_used",
                            "invalid_source_ids",
                            "contains source IDs not present in the source packet",
                            missing_ids,
                        )
                    )

    if "uncertainty_notes" in record:
        validate_string_list(errors, "uncertainty_notes", record["uncertainty_notes"])

    return {"row_number": row_number, "event_id": event_id, "errors": errors}


def build_report(
    input_path: Path,
    input_format: str,
    rows: list[tuple[int, Any]],
    load_errors: list[dict[str, Any]],
    row_results: list[dict[str, Any]],
    report_path: Path,
    source_packets_path: Path | None,
    source_packet_index: dict[str, set[str]] | None,
    source_packet_warnings: list[dict[str, Any]],
    source_packet_file_count: int,
) -> dict[str, Any]:
    invalid_rows = [result for result in row_results if result["errors"]]
    error_counts_by_field: Counter[str] = Counter()
    error_counts_by_code: Counter[str] = Counter()
    total_row_errors = 0
    for result in invalid_rows:
        for error in result["errors"]:
            error_counts_by_field[error["field"]] += 1
            error_counts_by_code[error["code"]] += 1
            total_row_errors += 1

    summary = {
        "total_rows": len(rows),
        "valid_rows": len(rows) - len(invalid_rows),
        "invalid_rows": len(invalid_rows),
        "total_row_errors": total_row_errors,
        "load_errors": len(load_errors),
        "source_packet_validation_enabled": source_packets_path is not None,
        "source_packet_files_scanned": source_packet_file_count,
        "source_packet_events_indexed": 0 if source_packet_index is None else len(source_packet_index),
        "source_packet_warnings": len(source_packet_warnings),
    }

    return {
        "generated_at_utc": utc_now_iso(),
        "input_path": str(input_path),
        "input_format": input_format,
        "report_path": str(report_path),
        "all_valid": summary["invalid_rows"] == 0 and summary["load_errors"] == 0,
        "summary": summary,
        "schema": {
            "required_fields": list(REQUIRED_FIELDS),
            "allowed_treatments": sorted(TREATMENT_FAMILIES),
            "allowed_profiles": sorted(PROFILE_GROUP_BY_PROFILE),
            "allowed_profile_groups": list(ALLOWED_PROFILE_GROUPS),
            "allowed_model_families": list(ALLOWED_MODEL_FAMILIES),
            "allowed_upstream_model_families": list(ALLOWED_UPSTREAM_MODEL_FAMILIES),
            "allowed_actions": list(ALLOWED_ACTIONS),
            "numeric_ranges": {
                "expected_return_5d": [-0.20, 0.20],
                "confidence": [0.0, 1.0],
                "action_strength": [-1.0, 1.0],
            },
            "seed_rules": {
                "representation_seed": {
                    "T1_B0": sorted(ALLOWED_DETERMINISTIC_REPRESENTATION_SEEDS),
                    "T2_T3_T4": sorted(ALLOWED_NON_T1_REPRESENTATION_SEEDS),
                },
                "decision_seed": sorted(ALLOWED_DECISION_SEEDS),
            },
            "list_rules": {
                "key_reasons": {"min_items": 2, "max_items": 5},
                "evidence_used": {"min_items": 1},
            },
        },
        "error_counts_by_field": dict(sorted(error_counts_by_field.items())),
        "error_counts_by_code": dict(sorted(error_counts_by_code.items())),
        "load_errors": load_errors,
        "source_packet_validation": {
            "path": None if source_packets_path is None else str(source_packets_path),
            "files_scanned": source_packet_file_count,
            "events_indexed": 0 if source_packet_index is None else len(source_packet_index),
            "warnings": source_packet_warnings,
        },
        "invalid_rows": invalid_rows,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(
        f"validated {summary['total_rows']} rows from {report['input_path']} "
        f"({report['input_format']})"
    )
    print(
        f"valid={summary['valid_rows']} invalid={summary['invalid_rows']} "
        f"row_errors={summary['total_row_errors']} load_errors={summary['load_errors']}"
    )
    if summary["source_packet_validation_enabled"]:
        print(
            "source_packets="
            f"{summary['source_packet_events_indexed']} events "
            f"across {summary['source_packet_files_scanned']} files "
            f"(warnings={summary['source_packet_warnings']})"
        )
    print(f"report={report['report_path']}")


def build_fatal_report(
    input_path: Path,
    report_path: Path,
    code: str,
    message: str,
    source_packets_path: Path | None,
) -> dict[str, Any]:
    return {
        "generated_at_utc": utc_now_iso(),
        "input_path": str(input_path),
        "input_format": "unknown",
        "report_path": str(report_path),
        "all_valid": False,
        "summary": {
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "total_row_errors": 0,
            "load_errors": 1,
            "source_packet_validation_enabled": source_packets_path is not None,
            "source_packet_files_scanned": 0,
            "source_packet_events_indexed": 0,
            "source_packet_warnings": 0,
        },
        "fatal_error": {"code": code, "message": message},
        "load_errors": [{"code": code, "message": message}],
        "source_packet_validation": {
            "path": None if source_packets_path is None else str(source_packets_path),
            "files_scanned": 0,
            "events_indexed": 0,
            "warnings": [],
        },
        "invalid_rows": [],
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    report_path = Path(args.report) if args.report else default_report_path(input_path)
    source_packets_path = None if args.source_packets is None else Path(args.source_packets)

    try:
        rows, load_errors, input_format = load_input_records(input_path)
    except Exception as exc:
        report = build_fatal_report(
            input_path,
            report_path,
            "input_load_failed",
            str(exc),
            source_packets_path,
        )
        write_report(report_path, report)
        print(f"validation failed: {exc}", file=sys.stderr)
        print(f"report={report_path}")
        return 2

    source_packet_index: dict[str, set[str]] | None = None
    source_packet_warnings: list[dict[str, Any]] = []
    source_packet_file_count = 0
    if source_packets_path is not None:
        try:
            source_packet_index, source_packet_warnings, source_packet_file_count = (
                load_source_packet_index(source_packets_path)
            )
        except Exception as exc:
            report = build_fatal_report(
                input_path,
                report_path,
                "source_packet_load_failed",
                str(exc),
                source_packets_path,
            )
            write_report(report_path, report)
            print(f"validation failed: {exc}", file=sys.stderr)
            print(f"report={report_path}")
            return 2

    row_results = [
        validate_record(row_number, record, source_packet_index)
        for row_number, record in rows
    ]
    report = build_report(
        input_path=input_path,
        input_format=input_format,
        rows=rows,
        load_errors=load_errors,
        row_results=row_results,
        report_path=report_path,
        source_packets_path=source_packets_path,
        source_packet_index=source_packet_index,
        source_packet_warnings=source_packet_warnings,
        source_packet_file_count=source_packet_file_count,
    )
    write_report(report_path, report)
    print_summary(report)
    return 0 if report["all_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
