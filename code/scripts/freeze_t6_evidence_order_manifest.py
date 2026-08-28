#!/usr/bin/env python3
"""Freeze and verify the one-per-event T6 order-randomization manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "clean_strict_predata_2026_main" / "t6_evidence_order_randomization_20260711"
REQUIRED_FIELDS = (
    "event_id", "treatment", "permutation_version", "master_seed",
    "original_evidence_ids", "permuted_evidence_ids", "unit_count",
    "original_ordered_units_sha256", "permuted_ordered_units_sha256",
    "unordered_units_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_ROOT / "t6_order_randomization_manifest.jsonl"))
    parser.add_argument("--freeze-record", default=str(DEFAULT_ROOT / "t6_order_randomization_freeze.json"))
    parser.add_argument("--expected-events", type=int, default=94)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def manifest_summary(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if not rows:
        raise ValueError("T6 manifest is empty")
    event_ids: list[str] = []
    for row in rows:
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"T6 manifest missing fields: {missing}")
        if row["treatment"] != "T6_canonical_evidence_order_randomized":
            raise ValueError("unexpected T6 treatment")
        original, permuted = row["original_evidence_ids"], row["permuted_evidence_ids"]
        if not isinstance(original, list) or not isinstance(permuted, list) or original == permuted:
            raise ValueError(f"{row.get('event_id')} lacks a non-identity order change")
        if len(original) != int(row["unit_count"]) or sorted(original) != sorted(permuted):
            raise ValueError(f"{row.get('event_id')} changed evidence coverage")
        event_ids.append(str(row["event_id"]))
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("duplicate T6 events")
    return {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "event_count": len(rows),
        "event_ids": sorted(event_ids),
        "master_seed": rows[0]["master_seed"],
        "permutation_version": rows[0]["permutation_version"],
    }


def main() -> int:
    args = parse_args()
    manifest, freeze_record = resolve(args.manifest), resolve(args.freeze_record)
    summary = manifest_summary(manifest)
    if summary["event_count"] != args.expected_events:
        raise ValueError(f"expected {args.expected_events} events; found {summary['event_count']}")
    if args.verify_only:
        record = json.loads(freeze_record.read_text(encoding="utf-8"))
        for key, value in summary.items():
            if record.get(key) != value:
                raise ValueError(f"freeze record mismatch for {key}")
        print("T6 manifest freeze verification: OK")
        return 0
    if freeze_record.exists():
        raise FileExistsError(f"freeze record exists: {freeze_record}")
    record = {
        "treatment": "T6_canonical_evidence_order_randomized",
        "frozen_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_revision": git_revision(),
        "manifest_path": str(manifest.relative_to(PROJECT_ROOT)),
        **summary,
    }
    freeze_record.write_text(json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
