#!/usr/bin/env python3
"""Audit masked source packets for residual identity-leakage clues.

This script is a static pre-provider gate. It does not call model APIs. It
scans agent-facing masked packet JSON/JSONL files and reports residual clues
that could let a model infer issuer or event identity before paid memory probes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "masking_leakage_audit"

MONTH_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)"
)
EXACT_DATE_PATTERN = re.compile(
    rf"\b{MONTH_PATTERN}\.?\s+\d{{1,2}}(?:,\s+\d{{4}})?\b",
    re.IGNORECASE,
)
EXCHANGE_TICKER_PATTERN = re.compile(
    r"\b(?:Nasdaq|NASDAQ|NYSE|NYSE American|AMEX|OTC|Cboe)\s*[:：]\s*"
    r"(?!Company\s+[A-Z]\b)[A-Z][A-Z.\-]{0,7}\b"
)
ACCESSION_PATTERN = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")
CIK_PATTERN = re.compile(r"\bCIK\s*[:#]?\s*\d{6,10}\b", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)[^\s<>)\"']+", re.IGNORECASE)
DOMAIN_PATTERN = re.compile(
    r"\b(?!Company\s+A\.com\b)[A-Za-z0-9][A-Za-z0-9.-]{1,}\."
    r"(?:com|net|org|io|ai|co|us|edu|gov)\b",
    re.IGNORECASE,
)
EXECUTIVE_NAME_BEFORE_TITLE = re.compile(
    r"\b(?P<name>[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,2})\s*,\s*"
    r"(?:chair|chairman|chairwoman|chairperson|ceo|cfo|coo|president|"
    r"chief\s+[a-z ]+officer|founder|general counsel)\b",
    re.IGNORECASE,
)
EXECUTIVE_NAME_AFTER_SAID = re.compile(
    r"\b(?:said|stated|commented|noted|added)\s+"
    r"(?P<name>[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,2})\b"
)
DISTINCTIVE_METRIC_PATTERN = re.compile(
    r"\b(?:ARR|annualized recurring revenue|remaining performance obligations|"
    r"\bRPO\b|same-store sales|comparable store sales|bookings|backlog|"
    r"net retention|gross merchandise value|GMV|annual recurring revenue)\b",
    re.IGNORECASE,
)
TRANSACTION_PATTERN = re.compile(
    r"\b(?:acquisition|merger|divestiture|spin-?off|tender offer|"
    r"regulatory approval|closing conditions)\b",
    re.IGNORECASE,
)
POST_EVENT_KEY_PATTERN = re.compile(
    r"(?:hidden_outcomes?|hidden_valence|post_event|market_reaction|"
    r"price_reaction|abnormal_return|car_)",
    re.IGNORECASE,
)
PLACE_PATTERN = re.compile(
    r"\b[A-Z][A-Z .'-]{2,},\s+(?:Ala|Alaska|Ariz|Ark|Calif|Colo|Conn|Del|"
    r"Fla|Ga|Ill|Ind|Iowa|Kan|Ky|La|Md|Mass|Mich|Minn|Mo|Nev|N\.J|N\.Y|"
    r"N\.C|Ohio|Okla|Ore|Pa|Tenn|Texas|Utah|Va|Wash|Wis)\.?\b"
)

TEXT_RULES = (
    ("exchange_ticker", "high", EXCHANGE_TICKER_PATTERN),
    ("accession_number", "high", ACCESSION_PATTERN),
    ("cik_reference", "high", CIK_PATTERN),
    ("email", "high", EMAIL_PATTERN),
    ("url", "medium", URL_PATTERN),
    ("domain", "medium", DOMAIN_PATTERN),
    ("executive_name_before_title", "medium", EXECUTIVE_NAME_BEFORE_TITLE),
    ("executive_name_after_said", "medium", EXECUTIVE_NAME_AFTER_SAID),
    ("exact_calendar_date", "medium", EXACT_DATE_PATTERN),
    ("specific_place", "medium", PLACE_PATTERN),
    ("distinctive_metric", "low", DISTINCTIVE_METRIC_PATTERN),
    ("transaction_language", "low", TRANSACTION_PATTERN),
)

FIELD_NAME_RULES = {
    "cik": "direct_identifier_field",
    "accession_number": "direct_identifier_field",
    "ticker": "direct_identifier_field",
    "issuer_ticker": "direct_identifier_field",
    "company_name": "direct_identifier_field",
    "issuer_name": "direct_identifier_field",
}

CSV_COLUMNS = [
    "event_id",
    "source_path",
    "risk_level",
    "finding_count",
    "high_count",
    "medium_count",
    "low_count",
    "categories",
    "example_count",
    "examples",
]
RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}
NON_PERSON_MATCH_TOKENS = {
    "chief",
    "executive",
    "financial",
    "operating",
    "officer",
    "president",
}


@dataclass
class Finding:
    event_id: str
    source_path: str
    json_path: str
    category: str
    severity: str
    match_text: str
    context: str


@dataclass
class PacketAudit:
    event_id: str
    source_path: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        if not self.findings:
            return "none"
        return max((finding.severity for finding in self.findings), key=RISK_ORDER.get)

    def severity_counts(self) -> Counter[str]:
        return Counter(finding.severity for finding in self.findings)

    def categories(self) -> list[str]:
        return sorted({finding.category for finding in self.findings})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Masked packet JSON/JSONL file(s) or directories",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--max-examples-per-event",
        type=int,
        default=8,
        help="Maximum examples stored in the per-event CSV example field.",
    )
    parser.add_argument(
        "--fail-on-high-risk",
        action="store_true",
        help="Return exit code 1 if any packet has high residual leakage risk.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def risk_max(values: Iterable[str]) -> str:
    return max(values, key=RISK_ORDER.get, default="none")


def compact_context(value: str, start: int, end: int, radius: int = 80) -> str:
    left = max(0, start - radius)
    right = min(len(value), end + radius)
    context = value[left:right]
    return " ".join(context.split())


def truncate(value: str, max_chars: int = 220) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def append_finding(
    findings: list[Finding],
    *,
    event_id: str,
    source_path: str,
    json_path: str,
    category: str,
    severity: str,
    match_text: str,
    context: str,
) -> None:
    findings.append(
        Finding(
            event_id=event_id,
            source_path=source_path,
            json_path=json_path,
            category=category,
            severity=severity,
            match_text=truncate(match_text, 160),
            context=truncate(context, 260),
        )
    )


def iter_json_text_values(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from iter_json_text_values(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_json_text_values(item, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def iter_json_fields(value: Any, path: str = "$") -> Iterable[tuple[str, Any, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            yield str(key), item, child_path
            yield from iter_json_fields(item, child_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_json_fields(item, f"{path}[{index}]")


def load_json_file(path: Path) -> list[tuple[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [(str(path.resolve()), payload)]
    if isinstance(payload, list):
        rows = []
        for index, item in enumerate(payload):
            if isinstance(item, dict):
                rows.append((f"{path.resolve()}#L{index + 1}", item))
        return rows
    return []


def load_jsonl_file(path: Path) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                rows.append((f"{path.resolve()}#L{line_number}", payload))
    return rows


def input_paths(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            paths.append(path.resolve())
            continue
        json_files = sorted(path.rglob("*.json"))
        jsonl_files = sorted(path.rglob("*.jsonl"))
        paths.extend(candidate.resolve() for candidate in json_files + jsonl_files)
    unique = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def is_packet_record(record: dict[str, Any]) -> bool:
    if not isinstance(record.get("event_id"), str):
        return False
    return "source_units" in record or "filing_metadata" in record or "xbrl_facts" in record


def scan_text_fields(record: dict[str, Any], source_path: str) -> list[Finding]:
    event_id = str(record.get("event_id", "unknown")).strip() or "unknown"
    findings: list[Finding] = []
    for json_path, text in iter_json_text_values(record):
        if not text.strip():
            continue
        normalized = " ".join(text.split())
        for category, severity, pattern in TEXT_RULES:
            for match in pattern.finditer(normalized):
                if category.startswith("executive_name"):
                    tokens = {token.lower().strip(".,") for token in match.group(0).split()}
                    if tokens.intersection(NON_PERSON_MATCH_TOKENS):
                        continue
                append_finding(
                    findings,
                    event_id=event_id,
                    source_path=source_path,
                    json_path=json_path,
                    category=category,
                    severity=severity,
                    match_text=match.group(0),
                    context=compact_context(normalized, match.start(), match.end()),
                )
    return findings


def scan_field_values(record: dict[str, Any], source_path: str) -> list[Finding]:
    event_id = str(record.get("event_id", "unknown")).strip() or "unknown"
    findings: list[Finding] = []
    for key, value, json_path in iter_json_fields(record):
        key_lower = key.lower()
        if POST_EVENT_KEY_PATTERN.search(key_lower):
            append_finding(
                findings,
                event_id=event_id,
                source_path=source_path,
                json_path=json_path,
                category="post_event_field_present",
                severity="high",
                match_text=key,
                context=f"{key} is present in an agent-facing packet",
            )
            continue
        category = FIELD_NAME_RULES.get(key_lower)
        if category is None or value in (None, "", [], {}):
            continue
        severity = "medium" if key_lower == "masked_company" else "high"
        append_finding(
            findings,
            event_id=event_id,
            source_path=source_path,
            json_path=json_path,
            category=category,
            severity=severity,
            match_text=str(value),
            context=f"{key} has non-empty value",
        )
    return findings


def audit_record(source_path: str, record: dict[str, Any]) -> PacketAudit:
    event_id = str(record.get("event_id", "unknown")).strip() or "unknown"
    audit = PacketAudit(event_id=event_id, source_path=source_path)
    audit.findings.extend(scan_field_values(record, source_path))
    audit.findings.extend(scan_text_fields(record, source_path))
    return audit


def load_records(paths: list[Path]) -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    for path in paths:
        try:
            loaded = load_jsonl_file(path) if path.suffix.lower() == ".jsonl" else load_json_file(path)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}: {exc}") from exc
        for source_path, record in loaded:
            if is_packet_record(record):
                records.append((source_path, record))
    return records


def write_outputs(
    audits: list[PacketAudit],
    output_dir: Path,
    max_examples_per_event: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_path = output_dir / "masking_leakage_findings.jsonl"
    events_path = output_dir / "masking_leakage_events.csv"
    summary_path = output_dir / "run_summary.json"

    with findings_path.open("w", encoding="utf-8") as handle:
        for audit in audits:
            for finding in audit.findings:
                handle.write(json.dumps(finding.__dict__, ensure_ascii=True, sort_keys=True))
                handle.write("\n")

    risk_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    with events_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for audit in audits:
            counts = audit.severity_counts()
            risk_counts[audit.risk_level] += 1
            severity_counts.update(counts)
            category_counts.update(finding.category for finding in audit.findings)
            examples = [
                f"{finding.severity}:{finding.category}:{finding.match_text} @ {finding.json_path}"
                for finding in audit.findings[:max_examples_per_event]
            ]
            writer.writerow(
                {
                    "event_id": audit.event_id,
                    "source_path": audit.source_path,
                    "risk_level": audit.risk_level,
                    "finding_count": len(audit.findings),
                    "high_count": counts.get("high", 0),
                    "medium_count": counts.get("medium", 0),
                    "low_count": counts.get("low", 0),
                    "categories": ";".join(audit.categories()),
                    "example_count": len(examples),
                    "examples": " | ".join(examples),
                }
            )

    summary = {
        "schema_version": "masking_leakage_audit_summary_v1",
        "created_at_utc": utc_now_iso(),
        "packet_count": len(audits),
        "finding_count": sum(len(audit.findings) for audit in audits),
        "max_risk": risk_max(audit.risk_level for audit in audits),
        "risk_counts": dict(sorted(risk_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "category_counts": dict(category_counts.most_common()),
        "outputs": {
            "events_csv": str(events_path.resolve()),
            "findings_jsonl": str(findings_path.resolve()),
            "summary_json": str(summary_path.resolve()),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    if args.max_examples_per_event < 0:
        raise ValueError("--max-examples-per-event must be >= 0")

    paths = input_paths(args.inputs)
    records = load_records(paths)
    audits = [audit_record(source_path, record) for source_path, record in records]
    summary = write_outputs(
        audits=audits,
        output_dir=Path(args.output_dir),
        max_examples_per_event=args.max_examples_per_event,
    )

    print("masking leakage audit summary:")
    print(f"  packet_count: {summary['packet_count']}")
    print(f"  finding_count: {summary['finding_count']}")
    print(f"  max_risk: {summary['max_risk']}")
    print(f"  risk_counts: {summary['risk_counts']}")
    print(f"  output_dir: {Path(args.output_dir).resolve()}")
    if args.fail_on_high_risk and summary["max_risk"] == "high":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
