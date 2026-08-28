#!/usr/bin/env python3
"""Build entity-visible, outcome-blind agent packets from masked packet artifacts.

The current source-packet builder masks issuer identity for the original
identity-ablation design. This transformer restores allowed issuer metadata from
the packet manifest while keeping hidden outcomes in sidecar files only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_VISIBILITY_POLICY = "entity_visible_outcome_blind"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-packets-dir", required=True)
    parser.add_argument("--packet-manifest-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {
            row["event_id"]: row
            for row in csv.DictReader(handle)
            if row.get("event_id")
        }
    if not rows:
        raise ValueError(f"no manifest rows found in {path}")
    return rows


def load_packet(path: Path) -> dict[str, Any]:
    packet = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(packet, dict):
        raise ValueError(f"packet must be a JSON object: {path}")
    return packet


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def csv_bool(value: Any) -> bool | None:
    text = string_or_none(value)
    if text is None:
        return None
    normalized = text.lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def unmask_text(text: str, manifest: dict[str, str]) -> str:
    masked_company = string_or_none(manifest.get("masked_company"))
    company = string_or_none(manifest.get("company"))
    ticker = string_or_none(manifest.get("ticker"))
    accepted_at = string_or_none(manifest.get("accepted_at_et"))

    updated = text
    if masked_company and ticker:
        exchange_pattern = re.compile(
            rf"\((NYSE|NASDAQ|Nasdaq|NasdaqGS|NYSE American)\s*:\s*{re.escape(masked_company)}\)"
        )
        updated = exchange_pattern.sub(lambda match: f"({match.group(1)}:{ticker})", updated)
    if masked_company and company:
        updated = updated.replace(masked_company, company)
    if accepted_at:
        updated = updated.replace("masked_timestamp", accepted_at)
    return updated


def transform_source_units(
    source_units: Any,
    manifest: dict[str, str],
) -> list[dict[str, Any]]:
    if not isinstance(source_units, list):
        return []
    transformed = []
    for item in source_units:
        if not isinstance(item, dict):
            continue
        updated = dict(item)
        text = updated.get("text")
        if isinstance(text, str):
            updated["text"] = unmask_text(text, manifest)
        heading = updated.get("heading")
        if isinstance(heading, str):
            updated["heading"] = unmask_text(heading, manifest)
        transformed.append(updated)
    return transformed


def transform_packet(packet: dict[str, Any], manifest: dict[str, str]) -> dict[str, Any]:
    filing_metadata = packet.get("filing_metadata")
    if not isinstance(filing_metadata, dict):
        filing_metadata = {}

    after_close = csv_bool(manifest.get("after_close"))
    if after_close is None:
        after_close = filing_metadata.get("after_close")

    output = {
        "event_id": packet["event_id"],
        "visibility_policy": DEFAULT_VISIBILITY_POLICY,
        "identity_visibility": "visible",
        "company_name": string_or_none(manifest.get("company")),
        "ticker": string_or_none(manifest.get("ticker")),
        "company_alias": string_or_none(manifest.get("company")),
        "masked_company": string_or_none(manifest.get("masked_company")),
        "sector": string_or_none(manifest.get("sector")) or packet.get("sector"),
        "market_cap_group": string_or_none(manifest.get("market_cap_group"))
        or packet.get("market_cap_group"),
        "fiscal_period": string_or_none(manifest.get("fiscal_period"))
        or packet.get("fiscal_period"),
        "event_date": string_or_none(manifest.get("event_date")),
        "filing_metadata": {
            "cik": string_or_none(manifest.get("cik")),
            "accession_number": string_or_none(manifest.get("accession_number")),
            "form_type": string_or_none(manifest.get("form_type"))
            or filing_metadata.get("form_type"),
            "accepted_at_et": string_or_none(manifest.get("accepted_at_et")),
            "event_date": string_or_none(manifest.get("event_date")),
            "after_close": after_close,
            "items": filing_metadata.get("items"),
        },
        "source_units": transform_source_units(packet.get("source_units"), manifest),
        "xbrl_facts": packet.get("xbrl_facts", []),
        "pre_event_price_context": packet.get("pre_event_price_context", {}),
        "outcome_blind": True,
    }
    return output


def build_manifest_row(
    packet_path: Path,
    output_path: Path,
    packet: dict[str, Any],
    manifest: dict[str, str],
) -> dict[str, Any]:
    return {
        "event_id": packet["event_id"],
        "company_name": manifest.get("company", ""),
        "ticker": manifest.get("ticker", ""),
        "event_date": manifest.get("event_date", ""),
        "accepted_at_et": manifest.get("accepted_at_et", ""),
        "visibility_policy": DEFAULT_VISIBILITY_POLICY,
        "input_packet_path": str(packet_path),
        "output_packet_path": str(output_path),
    }


def main() -> int:
    args = parse_args()
    source_packets_dir = Path(args.source_packets_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_by_event = load_manifest(Path(args.packet_manifest_csv))

    rows = []
    for packet_path in sorted(source_packets_dir.glob("*.json")):
        packet = load_packet(packet_path)
        event_id = string_or_none(packet.get("event_id"))
        if event_id is None:
            continue
        manifest = manifest_by_event.get(event_id)
        if manifest is None:
            raise ValueError(f"missing manifest metadata for {event_id}")
        transformed = transform_packet(packet, manifest)
        output_path = output_dir / f"{event_id}.json"
        write_json(output_path, transformed)
        rows.append(build_manifest_row(packet_path, output_path, transformed, manifest))

    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, {"rows": rows})
    print(f"wrote {len(rows)} entity-visible packets to {output_dir}")
    print(f"wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
