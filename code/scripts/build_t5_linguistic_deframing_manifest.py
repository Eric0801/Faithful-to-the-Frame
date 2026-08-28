#!/usr/bin/env python3
"""Build an auditable candidate manifest for T5 linguistic de-framing.

The manifest is a paired, row-level review artifact over the frozen,
entity-visible canonical evidence banks. It never calls a model or external
service. Candidate detection is intentionally high precision: all unmatched
claims remain byte-identical, and automatic edits are limited to a small set
of deterministic sentence-opening speaker-stance phrases.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_BANKS_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "evidence_banks_entity_visible"
)
DEFAULT_T5_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "t5_linguistic_deframing_20260710"
)
DEFAULT_OUTPUT_JSONL = DEFAULT_T5_DIR / "t5_transformation_manifest.jsonl"
DEFAULT_SUMMARY_JSON = DEFAULT_T5_DIR / "t5_transformation_manifest_summary.json"

FRAME_SELF_EVALUATION = "self_evaluation"
FRAME_PROMOTIONAL_SUPERLATIVE = "promotional_superlative"
FRAME_RHETORICAL_INTENSIFIER = "rhetorical_intensifier"
FRAME_SPEAKER_STANCE = "speaker_stance_or_self_credit"

FRAME_CATEGORIES = (
    FRAME_SELF_EVALUATION,
    FRAME_PROMOTIONAL_SUPERLATIVE,
    FRAME_RHETORICAL_INTENSIFIER,
    FRAME_SPEAKER_STANCE,
)

# These patterns flag local language for human review. They do not establish
# that a claim is false, nor do they authorize a free-form rewrite.
CANDIDATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        FRAME_SELF_EVALUATION,
        re.compile(
            r"\b(?:we|our company|the company|management)\s+"
            r"(?:delivered|achieved|produced)\s+(?:a|an|another)\s+"
            r"(?:strong|solid|excellent|outstanding|exceptional|great)\s+"
            r"(?:quarter|year|result(?:s)?|performance)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        FRAME_SELF_EVALUATION,
        re.compile(
            r"\b(?:a|an|another)\s+"
            r"(?:strong|solid|excellent|outstanding|exceptional|great)\s+"
            r"(?:quarter|year|result(?:s)?|performance)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        FRAME_PROMOTIONAL_SUPERLATIVE,
        re.compile(
            r"\b(?:record-breaking|best-ever|industry-leading|world-class|"
            r"unmatched|unparalleled)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        FRAME_RHETORICAL_INTENSIFIER,
        re.compile(
            r"\b(?:very|extremely|exceptionally|particularly|highly)\s+"
            r"(?:strong|solid|robust|successful|positive|encouraging)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        FRAME_SPEAKER_STANCE,
        re.compile(
            r"\b(?:we|management|the company|our team)\s+"
            r"(?:are|were|remain|remained|continue to be|continued to be)\s+"
            r"(?:very\s+)?(?:pleased|proud|excited|delighted|confident|optimistic)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        FRAME_SPEAKER_STANCE,
        re.compile(
            r"\b(?:we|our team|management)\s+"
            r"(?:executed|delivered|achieved)\s+(?:well|successfully|strongly)\b",
            flags=re.IGNORECASE,
        ),
    ),
)

# The only automatic edit removes an opening subjective reporting preface and
# leaves every following character intact. All other candidates require review.
SAFE_OPENING_STANCE_RE = re.compile(
    r"^(?P<prefix>\s*(?:(?:[-*]|\u2022)\s+)?)"
    r"(?P<stance>"
    r"(?:we|management|the company)\s+(?:are|were)\s+"
    r"(?:pleased|proud|excited|delighted)\s+to\s+"
    r"(?:report|announce|share)\s+that\s+"
    r")",
    flags=re.IGNORECASE,
)

PROTECTED_TOKEN_RE = re.compile(
    r"(?:[$EURGBP]+\s?\d[\d,]*(?:\.\d+)?(?:%|[BMK])?|"
    r"\b\d[\d,]*(?:\.\d+)?%?|"
    r"\b(?:no|not|never|without|neither|nor|cannot|can't|didn't|doesn't|"
    r"wasn't|weren't|isn't|aren't|won't|wouldn't|shouldn't)\b|"
    r"\b(?:vs\.?|versus|compared\s+(?:with|to)|higher|lower|more|less|"
    r"above|below|than|guidance|outlook|risk(?:s)?|uncertainty|uncertain)\b)",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for manifest construction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-banks-dir",
        default=str(DEFAULT_EVIDENCE_BANKS_DIR),
        help="Directory containing entity-visible evt_*.json evidence banks.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(DEFAULT_OUTPUT_JSONL),
        help="Destination JSONL path for one candidate record per evidence unit.",
    )
    parser.add_argument(
        "--summary-json",
        default=str(DEFAULT_SUMMARY_JSON),
        help="Destination JSON path for aggregate candidate counts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output JSONL or summary JSON.",
    )
    return parser.parse_args()


def compact_whitespace(value: Any) -> str:
    """Return a single-line string without changing the original claim."""
    return " ".join(str(value if value is not None else "").split())


def read_json(path: Path) -> dict[str, Any]:
    """Read one evidence bank and require a top-level JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def project_relative(path: Path) -> str:
    """Return a stable project-relative path where possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def candidate_frame_spans(claim: str) -> list[dict[str, Any]]:
    """Find and merge pre-specified high-precision candidate framing spans."""
    categories_by_span: dict[tuple[int, int, str], set[str]] = {}
    for category, pattern in CANDIDATE_PATTERNS:
        for match in pattern.finditer(claim):
            key = (match.start(), match.end(), match.group(0))
            categories_by_span.setdefault(key, set()).add(category)

    merged: list[tuple[int, int, str, set[str]]] = []
    for (start, end, text), categories in sorted(
        categories_by_span.items(), key=lambda item: (item[0][0], -item[0][1])
    ):
        containing_span = next(
            (
                index
                for index, (outer_start, outer_end, _, _) in enumerate(merged)
                if outer_start <= start and end <= outer_end
            ),
            None,
        )
        if containing_span is not None:
            merged[containing_span][3].update(categories)
            continue
        merged.append((start, end, text, set(categories)))

    spans: list[dict[str, Any]] = []
    for start, end, text, categories in merged:
        spans.append(
            {
                "text": text,
                "start_char": start,
                "end_char": end,
                "categories": sorted(categories),
            }
        )
    return spans


def protected_tokens(claim: str) -> list[str]:
    """Collect protected factual markers used to verify automatic local edits."""
    return [match.group(0) for match in PROTECTED_TOKEN_RE.finditer(claim)]


def capitalize_first_alpha(text: str) -> str:
    """Capitalize only the first alphabetic character after a safe prefix edit."""
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1 :]
    return text


def deterministic_safe_edit(claim: str) -> tuple[str, str | None]:
    """Apply the sole permitted local edit or return the original claim.

    The fixed opening phrase contains speaker stance but no operational claim.
    The edit is accepted only when every protected factual marker remains in
    the resulting claim in the same sequence.
    """
    match = SAFE_OPENING_STANCE_RE.match(claim)
    if match is None:
        return claim, None

    proposed = capitalize_first_alpha(match.group("prefix") + claim[match.end() :])
    if not compact_whitespace(proposed):
        return claim, None
    if protected_tokens(claim) != protected_tokens(proposed):
        return claim, None
    return proposed, "remove_opening_speaker_stance"


def clean_source_ids(value: Any) -> list[str]:
    """Preserve source IDs as strings in their canonical evidence-bank order."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def manifest_record(event_id: str, unit: dict[str, Any]) -> dict[str, Any]:
    """Build one auditable T5 candidate record from a canonical evidence unit."""
    original_claim = str(unit.get("claim") if unit.get("claim") is not None else "")
    spans = candidate_frame_spans(original_claim)
    proposed_t5_claim, automatic_edit = deterministic_safe_edit(original_claim)
    categories = sorted(
        {category for span in spans for category in span["categories"]}
    )
    review_status = "pending" if spans else "identity"
    return {
        "event_id": event_id,
        "evidence_id": compact_whitespace(unit.get("evidence_id")),
        "original_claim": original_claim,
        "category": unit.get("category"),
        "value": unit.get("value"),
        "source_ids": clean_source_ids(unit.get("source_ids")),
        "support_label": unit.get("support_label"),
        "candidate_frame_spans": spans,
        "candidate_frame_categories": categories,
        "proposed_t5_claim": proposed_t5_claim,
        "automatic_edit": automatic_edit,
        "approved": False,
        "review_status": review_status,
        "factual_change_forbidden": True,
    }


def iter_manifest_records(evidence_banks_dir: Path) -> Iterable[dict[str, Any]]:
    """Yield manifest records in stable event-file and evidence-bank order."""
    bank_paths = sorted(evidence_banks_dir.glob("evt_*.json"))
    if not bank_paths:
        raise ValueError(f"no evt_*.json evidence banks found in {evidence_banks_dir}")
    for bank_path in bank_paths:
        bank = read_json(bank_path)
        event_id = compact_whitespace(bank.get("event_id"))
        if not event_id:
            raise ValueError(f"{bank_path} is missing event_id")
        units = bank.get("evidence_units")
        if not isinstance(units, list):
            raise ValueError(f"{bank_path} evidence_units must be a list")
        for index, unit in enumerate(units, start=1):
            if not isinstance(unit, dict):
                raise ValueError(f"{bank_path} evidence_units[{index}] must be an object")
            yield manifest_record(event_id, unit)


def require_writable_outputs(
    output_jsonl: Path, summary_json: Path, overwrite: bool
) -> None:
    """Fail before writing when either requested output already exists."""
    existing = [path for path in (output_jsonl, summary_json) if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output exists: {rendered}; pass --overwrite")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write deterministic ASCII JSONL records."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic ASCII JSON with stable key ordering."""
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_summary(
    records: list[dict[str, Any]], evidence_banks_dir: Path, output_jsonl: Path
) -> dict[str, Any]:
    """Summarize review workload and automatic-edit coverage."""
    candidate_category_counts: Counter[str] = Counter()
    for record in records:
        candidate_category_counts.update(record["candidate_frame_categories"])
    event_ids = sorted({str(record["event_id"]) for record in records})
    pending_records = [record for record in records if record["review_status"] == "pending"]
    automatic_edits = [record for record in records if record["automatic_edit"]]
    return {
        "treatment": "T5_linguistic_deframing_candidate_manifest",
        "detector_scope": list(FRAME_CATEGORIES),
        "evidence_banks_dir": project_relative(evidence_banks_dir),
        "output_jsonl": project_relative(output_jsonl),
        "events": len(event_ids),
        "event_ids": event_ids,
        "total_evidence_units": len(records),
        "identity_records": len(records) - len(pending_records),
        "pending_review_records": len(pending_records),
        "automatic_edit_records": len(automatic_edits),
        "candidate_category_counts": dict(sorted(candidate_category_counts.items())),
        "factual_change_forbidden": True,
        "automatic_edit_policy": "opening speaker stance only; all other candidates remain pending",
    }


def main() -> int:
    """Build the T5 linguistic de-framing candidate manifest."""
    args = parse_args()
    evidence_banks_dir = Path(args.evidence_banks_dir).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    summary_json = Path(args.summary_json).expanduser().resolve()
    if not evidence_banks_dir.is_dir():
        raise NotADirectoryError(f"evidence banks directory does not exist: {evidence_banks_dir}")
    require_writable_outputs(output_jsonl, summary_json, args.overwrite)

    records = list(iter_manifest_records(evidence_banks_dir))
    if not records:
        raise ValueError(f"no evidence units found in {evidence_banks_dir}")
    summary = build_summary(records, evidence_banks_dir, output_jsonl)
    write_jsonl(output_jsonl, records)
    write_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    print(f"output_jsonl={output_jsonl}")
    print(f"summary_json={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
