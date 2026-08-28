#!/usr/bin/env python3
"""Apply the author-reviewed T5 adjudication decisions deterministically."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "t5_linguistic_deframing_20260710"
)
DEFAULT_MANIFEST = DEFAULT_ROOT / "t5_transformation_manifest.jsonl"
DEFAULT_SUMMARY = DEFAULT_ROOT / "t5_transformation_manifest_review_summary.json"


def key(event_id: str, evidence_id: str) -> tuple[str, str]:
    return event_id, evidence_id


APPROVALS: dict[tuple[str, str], dict[str, Any]] = {
    key("evt_0002", "E_SRC_007"): {
        "claim": "Chief Executive Officer commented, \u201cBusiness momentum accelerated further in the fourth quarter, with both sales and earnings significantly surpassing our expectations.\u201d",
        "spans": [("We are pleased to report that", ["speaker_stance_or_self_credit"])],
    },
    key("evt_0011", "E_SRC_011"): {
        "claim": "\u201cQ3 organic revenue growth was 6%, ahead of guidance,\u201d said Geoff Martha, Medtronic plc chairman and chief executive officer.",
        "spans": [
            ("another strong quarter", ["self_evaluation"]),
            ("demonstrating the strength of our portfolio", ["self_evaluation"]),
        ],
    },
    key("evt_0012", "E_SRC_016"): {
        "claim": "In the U.S., the company saw intentional and urgent treatment of patients with severe aortic stenosis, fueled by a large and growing body of evidence on the SAPIEN platform.",
        "spans": [("world-class", ["promotional_superlative"])],
    },
    key("evt_0046", "E_SRC_013"): {
        "claim": "\u201cFortinet's fourth-quarter billings were above the high end of guidance, driven by broad-based demand across its portfolio,\u201d said Ken Xie, Founder, Chairman and Chief Executive Officer of Fortinet, Inc.",
        "spans": [
            ("We are pleased with", ["speaker_stance_or_self_credit"]),
            ("strong finish to the year", ["self_evaluation"]),
            ("excellent fourth quarter", ["self_evaluation"]),
        ],
    },
    key("evt_0051", "E_SRC_004"): {
        "claim": "\u201cFirst-quarter financial performance was driven by broad-based growth across channels, regions and categories,\u201d said President and CEO of LEVI STRAUSS & CO.",
        "spans": [
            ("We delivered", ["self_evaluation"]),
            ("very strong", ["rhetorical_intensifier"]),
        ],
    },
    key("evt_0051", "E_SRC_005"): {
        "claim": "\u201cFirst-quarter revenue, margins and EPS were above guidance,\u201d said Harmit Singh, Chief Financial and Growth Officer of LEVI STRAUSS & CO.",
        "spans": [("We are pleased to report", ["speaker_stance_or_self_credit"])],
    },
    key("evt_0052", "E_SRC_016"): {
        "claim": "\u201cThe MOVADO GROUP INC brand recorded sales growth of 25% in its wholesale business, 18% at MOVADO GROUP INC.com, and 9% in its Company Stores,\u201d Mr. ...",
        "spans": [
            (
                "delivered a very strong quarter across our most important channels, led by",
                ["self_evaluation", "rhetorical_intensifier"],
            )
        ],
    },
    key("evt_0057", "E_SRC_005"): {
        "claim": "\u201cSeveral commercial initiatives are underway.\u201d",
        "spans": [
            (
                "We remain optimistic about our trajectory and have several exciting",
                ["speaker_stance_or_self_credit", "rhetorical_intensifier"],
            )
        ],
    },
    key("evt_0063", "E_SRC_009"): {
        "claim": "\u201cIn 2025 GATX CORP announced its largest-ever railcar acquisition,\u201d said Robert C. ...",
        "spans": [
            (
                "2025 was an exceptional year for GATX CORP, highlighted by strong financial results and the announcement of",
                ["self_evaluation"],
            )
        ],
    },
    key("evt_0084", "E_SRC_013"): {
        "claim": "\u201cThe fourth quarter concluded the year for The Times,\u201d said Meredith Kopit Levien, president and chief executive officer, The NEW YORK TIMES CO Company.",
        "spans": [
            ("another strong year", ["self_evaluation"]),
            (
                "our results demonstrated that our strategy continues to work as designed",
                ["self_evaluation"],
            ),
        ],
    },
    key("evt_0088", "E_SRC_009"): {
        "claim": "\u201cThe company reaffirmed its preliminary outlook, provided on its third-quarter earnings call, of 10% to 11% revenue growth in 2026 compared with 2025.\u201d",
        "spans": [
            (
                "With the momentum we are seeing, we are pleased to",
                ["speaker_stance_or_self_credit"],
            )
        ],
    },
    key("evt_0088", "E_SRC_012"): {
        "claim": "\u201cMatt Osberg will join Inspire Medical Systems, Inc.\u201d",
        "spans": [("We are very excited to have", ["speaker_stance_or_self_credit"])],
    },
    key("evt_0089", "E_SRC_013"): {
        "claim": "\u201cThe fourth quarter concluded a year of growth and expanded margins.\u201d",
        "spans": [
            ("a solid close to a strong", ["self_evaluation"]),
            ("strong overall execution", ["self_evaluation"]),
        ],
    },
}

REJECTION_REASONS = {
    key("evt_0001", "E_SRC_007"): "pure_self_evaluation_no_retainable_proposition",
    key("evt_0025", "E_SRC_003"): "residual_self_evaluation_after_local_removal",
    key("evt_0033", "E_SRC_003"): "protected_external_comparator",
    key("evt_0033", "E_SRC_008"): "protected_external_comparator",
    key("evt_0033", "E_SRC_009"): "protected_external_comparator",
    key("evt_0033", "E_SRC_010"): "protected_external_comparator",
    key("evt_0033", "E_SRC_013"): "protected_external_comparator",
    key("evt_0033", "E_SRC_014"): "protected_external_comparator",
    key("evt_0033", "E_SRC_015"): "protected_external_comparator",
    key("evt_0033", "E_SRC_035"): "protected_external_comparator",
    key("evt_0033", "E_SRC_045"): "legal_qualification",
    key("evt_0039", "E_SRC_004"): "no_retainable_operational_proposition",
    key("evt_0043", "E_SRC_015"): "framing_entwined_with_unquantified_claim",
    key("evt_0053", "E_SRC_005"): "pure_self_evaluation_no_retainable_proposition",
    key("evt_0060", "E_SRC_006"): "framing_entwined_with_comparator_and_forward_claims",
    key("evt_0068", "E_SRC_018"): "pure_forward_looking_speaker_confidence",
    key("evt_0075", "E_SRC_076"): "legal_qualification",
    key("evt_0078", "E_SRC_006"): "pure_self_evaluation_no_retainable_proposition",
    key("evt_0088", "E_SRC_008"): "residual_self_evaluation_after_local_removal",
    key("evt_0090", "E_SRC_012"): "stance_entwined_with_guidance_and_business_claims",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def audit_spans(claim: str, span_specs: list[tuple[str, list[str]]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for text, categories in span_specs:
        start = claim.index(text)
        spans.append(
            {
                "text": text,
                "start_char": start,
                "end_char": start + len(text),
                "categories": sorted(categories),
            }
        )
    return sorted(spans, key=lambda item: (int(item["start_char"]), int(item["end_char"])))


def parse_manifest_line(line: str) -> dict[str, Any]:
    """Accept a manually edited JSONL row with Python boolean literals.

    The reviewed manifest must be written back as strict JSON. This compatibility
    path prevents a single hand-entered ``True`` or ``False`` from blocking the
    author-approved deterministic adjudication pass.
    """
    normalized = re.sub(r"(:\s*)True(?=\s*[,}])", r"\1true", line)
    normalized = re.sub(r"(:\s*)False(?=\s*[,}])", r"\1false", normalized)
    return json.loads(normalized)


def main() -> int:
    args = parse_args()
    manifest_path = resolve(args.manifest)
    summary_path = resolve(args.summary_json)
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"summary exists: {summary_path}; pass --overwrite")
    rows = [
        parse_manifest_line(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidate_rows = {
        key(str(row["event_id"]), str(row["evidence_id"]))
        for row in rows
        if row.get("candidate_frame_categories")
    }
    decisions = set(APPROVALS) | set(REJECTION_REASONS)
    if candidate_rows != decisions:
        raise ValueError(
            "candidate/decision key mismatch: "
            f"missing={sorted(candidate_rows - decisions)}, extra={sorted(decisions - candidate_rows)}"
        )

    for row in rows:
        row_key = key(str(row["event_id"]), str(row["evidence_id"]))
        if row_key in APPROVALS:
            decision = APPROVALS[row_key]
            original = str(row["original_claim"])
            spans = audit_spans(original, decision["spans"])
            row["approved"] = True
            row["review_status"] = "approved"
            row["proposed_t5_claim"] = decision["claim"]
            row["candidate_frame_spans"] = spans
            row["candidate_frame_categories"] = sorted(
                {category for span in spans for category in span["categories"]}
            )
            row["adjudication_reason"] = "author_approved_fact_preserving_local_deframing"
        elif row_key in REJECTION_REASONS:
            row["approved"] = False
            row["review_status"] = "rejected"
            row["proposed_t5_claim"] = row["original_claim"]
            row["adjudication_reason"] = REJECTION_REASONS[row_key]

    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    approved_rows = [row for row in rows if row.get("approved")]
    status_counts = Counter(str(row.get("review_status")) for row in rows)
    summary = {
        "manifest": str(manifest_path),
        "approved_edit_count": len(approved_rows),
        "edit_bearing_event_count": len({row["event_id"] for row in approved_rows}),
        "edit_bearing_event_ids": sorted({row["event_id"] for row in approved_rows}),
        "review_status_counts": dict(sorted(status_counts.items())),
        "author_decision": "2026-07-10 adjudication queue review",
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
