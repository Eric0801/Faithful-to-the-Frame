#!/usr/bin/env python3
"""Audit agent-facing artifacts for hidden outcome and future-information leaks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT = "outcome_blindness_audit.csv"
RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}

FIELD_PATTERNS = (
    ("hidden_outcome_field", "high", re.compile(r"hidden|CAR_1_5|CAR_1_20|valence", re.I)),
    ("realized_return_field", "high", re.compile(r"post_event|market_reaction|price_reaction", re.I)),
)
TEXT_PATTERNS = (
    ("hidden_outcome_text", "high", re.compile(r"hidden[_ -]?(?:outcome|valence)|CAR_1_5|CAR_1_20", re.I)),
    ("realized_reaction_text", "high", re.compile(r"realized market reaction|post-event return|abnormal return", re.I)),
    ("future_source_text", "medium", re.compile(r"future filings?/news|later filings?|subsequent filing", re.I)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT)
    parser.add_argument("--fail-on-high-risk", action="store_true")
    return parser.parse_args()


def discover_files(inputs: Iterable[str]) -> list[Path]:
    files = []
    for item in inputs:
        path = Path(item)
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate
                for candidate in sorted(path.rglob("*"))
                if candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl"}
            )
        else:
            raise FileNotFoundError(path)
    return files


def iter_records(path: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    yield f"{path}#L{line_number}", payload
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        yield str(path), payload


def iter_fields(value: Any, path: str = "$") -> Iterable[tuple[str, Any, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            yield str(key), item, child_path
            yield from iter_fields(item, child_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_fields(item, f"{path}[{index}]")


def iter_text(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from iter_text(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_text(item, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def max_risk(risks: Iterable[str]) -> str:
    return max(risks, key=lambda risk: RISK_ORDER[risk], default="none")


def audit_record(source: str, record: dict[str, Any]) -> dict[str, Any]:
    findings = []
    event_id = str(record.get("event_id", ""))
    if source.endswith("manifest.json"):
        return {
            "event_id": event_id,
            "source": source,
            "risk": "none",
            "finding_count": 0,
            "categories": "",
            "examples": "",
        }
    for key, value, json_path in iter_fields(record):
        for category, severity, pattern in FIELD_PATTERNS:
            if pattern.search(key):
                findings.append((category, severity, json_path, key))
        if key in {"ret_5d", "ret_20d", "market_ret_20d"}:
            continue
        if key.startswith("ret_") or key.startswith("market_ret_"):
            findings.append(("post_event_return_field", "high", json_path, key))
    for json_path, text in iter_text(record):
        if "outcome_blind" in json_path:
            continue
        if json_path.endswith(".prompt") and (
            "Do not mention post-event outcomes" in text
            or "post-event outcomes, realized market reactions" in text
        ):
            continue
        for category, severity, pattern in TEXT_PATTERNS:
            match = pattern.search(text)
            if match:
                findings.append((category, severity, json_path, match.group(0)))
    categories = sorted({finding[0] for finding in findings})
    examples = [
        f"{category}@{json_path}:{match_text}"
        for category, _, json_path, match_text in findings[:8]
    ]
    return {
        "event_id": event_id,
        "source": source,
        "risk": max_risk(finding[1] for finding in findings),
        "finding_count": len(findings),
        "categories": ";".join(categories),
        "examples": " || ".join(examples),
    }


def main() -> int:
    args = parse_args()
    rows = []
    for path in discover_files(args.inputs):
        for source, record in iter_records(path):
            row = audit_record(source, record)
            if row["finding_count"]:
                rows.append(row)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["event_id", "source", "risk", "finding_count", "categories", "examples"],
        )
        writer.writeheader()
        writer.writerows(rows)

    high_count = sum(1 for row in rows if row["risk"] == "high")
    print(f"findings: {len(rows)}")
    print(f"high_risk_rows: {high_count}")
    print(f"output_csv: {output}")
    if args.fail_on_high_risk and high_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
