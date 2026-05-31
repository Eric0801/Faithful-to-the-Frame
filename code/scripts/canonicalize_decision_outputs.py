#!/usr/bin/env python3
"""Deterministically canonicalize provider decision output fields."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUEST_KEY_FIELDS = (
    "event_id",
    "treatment",
    "profile",
    "profile_group",
    "model_family",
    "representation_seed",
    "decision_seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite decision-output JSONL fields using explicit alias maps. "
            "This is intended for deterministic provider-alias normalization, not "
            "free-form output cleanup."
        )
    )
    parser.add_argument("input", help="Input decision JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--summary",
        required=True,
        help="Normalization summary JSON path.",
    )
    parser.add_argument(
        "--model-family-alias",
        action="append",
        default=[],
        metavar="ALIAS=CANONICAL",
        help="Repeatable model_family alias mapping.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_aliases(raw_aliases: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for raw_alias in raw_aliases:
        alias, separator, canonical = raw_alias.partition("=")
        alias = alias.strip()
        canonical = canonical.strip()
        if not separator or not alias or not canonical:
            raise ValueError(f"alias must be ALIAS=CANONICAL, got {raw_alias!r}")
        aliases[alias] = canonical
    return aliases


def load_jsonl(
    path: Path,
) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        "line_number": line_number,
                        "code": "invalid_jsonl",
                        "message": f"{exc.msg} at column {exc.colno}",
                    }
                )
                continue
            if not isinstance(payload, dict):
                errors.append(
                    {
                        "line_number": line_number,
                        "code": "non_object_row",
                        "message": "decision output row must be a JSON object",
                    }
                )
                continue
            rows.append((line_number, payload))
    return rows, errors


def build_request_key(record: dict[str, Any]) -> str | None:
    values: list[str] = []
    for field in REQUEST_KEY_FIELDS:
        value = record.get(field)
        if value is None:
            return None
        values.append(str(value))
    return "|".join(values)


def canonicalize_record(
    record: dict[str, Any],
    *,
    model_family_aliases: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    output = dict(record)
    normalizations: dict[str, str] = {}

    model_family = output.get("model_family")
    if isinstance(model_family, str):
        canonical_model_family = model_family_aliases.get(model_family)
        if canonical_model_family is not None:
            output["model_family"] = canonical_model_family
            normalizations["model_family"] = f"{model_family} -> {canonical_model_family}"

    request_key = build_request_key(output)
    if request_key is not None and output.get("request_key") != request_key:
        previous = output.get("request_key")
        output["request_key"] = request_key
        normalizations["request_key"] = f"{previous} -> {request_key}"

    return output, normalizations


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    try:
        model_family_aliases = parse_aliases(args.model_family_alias)
        rows, load_errors = load_jsonl(input_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    normalized_rows: list[dict[str, Any]] = []
    normalization_counts: Counter[str] = Counter()
    normalization_examples: list[dict[str, Any]] = []
    for line_number, record in rows:
        normalized, normalizations = canonicalize_record(
            record,
            model_family_aliases=model_family_aliases,
        )
        normalized_rows.append(normalized)
        for field in normalizations:
            normalization_counts[field] += 1
        if normalizations and len(normalization_examples) < 20:
            normalization_examples.append(
                {
                    "line_number": line_number,
                    "event_id": record.get("event_id"),
                    "normalizations": normalizations,
                }
            )

    write_jsonl(output_path, normalized_rows)
    write_summary(
        summary_path,
        {
            "generated_at_utc": utc_now_iso(),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "summary_path": str(summary_path),
            "total_rows_loaded": len(rows),
            "load_errors": load_errors,
            "rows_written": len(normalized_rows),
            "model_family_aliases": model_family_aliases,
            "normalization_counts": dict(normalization_counts),
            "normalization_examples": normalization_examples,
        },
    )
    print(
        f"wrote {len(normalized_rows)} rows to {output_path} "
        f"with normalizations={dict(normalization_counts)}"
    )
    if load_errors:
        print(
            f"warning: skipped {len(load_errors)} invalid input rows",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
