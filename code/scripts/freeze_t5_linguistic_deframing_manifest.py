#!/usr/bin/env python3
"""Freeze and verify the reviewed manifest for the T5 paired control."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
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
DEFAULT_FREEZE_RECORD = DEFAULT_ROOT / "t5_manifest_freeze.json"
FINAL_REVIEW_STATUSES = {"approved", "rejected", "identity"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--freeze-record", default=str(DEFAULT_FREEZE_RECORD))
    parser.add_argument("--expected-approved-edits", type=int, required=True)
    parser.add_argument("--expected-edit-bearing-events", type=int, required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing freeze record rather than creating one.",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_manifest(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if not rows:
        raise ValueError("manifest is empty")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("manifest rows must be JSON objects")
    return rows, raw


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def manifest_summary(rows: list[dict[str, Any]], raw: bytes) -> dict[str, Any]:
    final_statuses = Counter(str(row.get("review_status", "")) for row in rows)
    pending = [row for row in rows if str(row.get("review_status", "")) not in FINAL_REVIEW_STATUSES]
    if pending:
        raise ValueError(f"manifest has {len(pending)} non-final review rows")

    approved = [row for row in rows if bool(row.get("approved"))]
    malformed = [
        row
        for row in approved
        if str(row.get("review_status")) != "approved"
        or str(row.get("proposed_t5_claim", "")) == str(row.get("original_claim", ""))
    ]
    if malformed:
        raise ValueError("approved rows must have review_status=approved and a changed claim")
    inconsistent = [
        row for row in rows if str(row.get("review_status")) == "approved" and not bool(row.get("approved"))
    ]
    if inconsistent:
        raise ValueError("approved review rows must set approved=true")

    event_ids = sorted({str(row["event_id"]) for row in approved})
    return {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_rows": len(rows),
        "approved_edit_count": len(approved),
        "edit_bearing_event_count": len(event_ids),
        "edit_bearing_event_ids": event_ids,
        "review_status_counts": dict(sorted(final_statuses.items())),
    }


def verify_record(record: dict[str, Any], summary: dict[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "manifest_sha256": summary["manifest_sha256"],
        "approved_edit_count": args.expected_approved_edits,
        "edit_bearing_event_count": args.expected_edit_bearing_events,
    }
    actual = {key: record.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"freeze record mismatch: expected {expected}, found {actual}")
    if record.get("edit_bearing_event_ids") != summary["edit_bearing_event_ids"]:
        raise ValueError("freeze record edit-bearing event IDs do not match manifest")


def main() -> int:
    args = parse_args()
    manifest = resolve(args.manifest)
    freeze_record = resolve(args.freeze_record)
    rows, raw = read_manifest(manifest)
    summary = manifest_summary(rows, raw)
    if summary["approved_edit_count"] != args.expected_approved_edits:
        raise ValueError(
            f"expected {args.expected_approved_edits} approved edits; "
            f"found {summary['approved_edit_count']}"
        )
    if summary["edit_bearing_event_count"] != args.expected_edit_bearing_events:
        raise ValueError(
            f"expected {args.expected_edit_bearing_events} edit-bearing events; "
            f"found {summary['edit_bearing_event_count']}"
        )

    if args.verify_only:
        verify_record(json.loads(freeze_record.read_text(encoding="utf-8")), summary, args)
        print("T5 manifest freeze verification: OK")
        return 0

    if freeze_record.exists():
        raise FileExistsError(f"freeze record exists: {freeze_record}; use --verify-only")
    record = {
        "treatment": "T5_linguistic_deframing",
        "frozen_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_revision": git_revision(),
        "manifest_path": project_relative(manifest),
        **summary,
    }
    freeze_record.parent.mkdir(parents=True, exist_ok=True)
    freeze_record.write_text(
        json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
