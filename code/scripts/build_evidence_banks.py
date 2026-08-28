#!/usr/bin/env python3
"""Build canonical evidence-unit banks from source packets.

This is a deterministic scaffold for the shared factual substrate used by
T2/T3/T4. It does not replace later source-support auditing or LLM-assisted
normalization, but it gives the pipeline a stable JSON shape and audit trail.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evidence_banks"

SECTION_TO_CATEGORY = {
    "revenue": "revenue",
    "sales": "revenue",
    "earnings": "earnings_or_eps",
    "eps": "earnings_or_eps",
    "margin": "margins",
    "profitability": "margins",
    "guidance": "guidance",
    "outlook": "guidance",
    "costs": "costs_or_expenses",
    "expenses": "costs_or_expenses",
    "risk": "risk_or_uncertainty",
    "uncertainty": "risk_or_uncertainty",
    "cash": "balance_sheet_or_cash",
    "balance": "balance_sheet_or_cash",
    "liquidity": "balance_sheet_or_cash",
    "price": "pre_event_price_context",
}

XBRL_TAG_TO_CATEGORY = {
    "revenues": "revenue",
    "salesrevenue": "revenue",
    "salesrevenuenet": "revenue",
    "netincome": "earnings_or_eps",
    "netincomeloss": "earnings_or_eps",
    "earningspersharediluted": "earnings_or_eps",
    "earningspersharebasic": "earnings_or_eps",
    "grossprofit": "margins",
    "operatingincomeloss": "margins",
    "cashandcashequivalentsatcarryingvalue": "balance_sheet_or_cash",
    "cashcashequivalentsandshortterminvestments": "balance_sheet_or_cash",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-packets-dir", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--max-quote-chars",
        type=int,
        default=280,
        help="Maximum length for source_quote fields",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_source_packets(path: Path) -> list[dict[str, Any]]:
    packets = []
    for json_path in sorted(path.glob("*.json")):
        packet = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(packet, dict) or not packet.get("event_id"):
            continue
        packet["_source_path"] = str(json_path)
        packets.append(packet)
    if not packets:
        raise ValueError(f"no JSON source packets found in {path}")
    return packets


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, max_chars: int) -> str:
    clean = normalize_space(text)
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def first_sentence(text: str) -> str:
    clean = normalize_space(text)
    if not clean:
        return ""
    match = re.search(r"(.+?[.!?])(?:\s|$)", clean)
    if match:
        return match.group(1).strip()
    return clean


def infer_category_from_section(section: Any) -> str:
    name = str(section or "").strip().lower()
    for key, category in SECTION_TO_CATEGORY.items():
        if key in name:
            return category
    return "other"


def infer_category_from_tag(tag: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", str(tag or "").lower())
    for key, category in XBRL_TAG_TO_CATEGORY.items():
        if key in normalized:
            return category
    return "other"


def build_claim_from_fact(fact: dict[str, Any]) -> str:
    tag = str(fact.get("tag", "fact")).strip()
    period = str(fact.get("period", "")).strip()
    value = fact.get("value")
    if period:
        return f"{tag} reported for {period}: {value}"
    return f"{tag}: {value}"


def build_source_unit_evidence(
    source_units: list[dict[str, Any]],
    max_quote_chars: int,
) -> list[dict[str, Any]]:
    evidence_units = []
    for idx, unit in enumerate(source_units, start=1):
        source_id = str(unit.get("source_id", f"S{idx}"))
        raw_text = str(unit.get("text", "")).strip()
        if not raw_text:
            continue
        claim = first_sentence(raw_text)
        evidence_units.append(
            {
                "evidence_id": f"E_SRC_{idx:03d}",
                "category": infer_category_from_section(unit.get("section")),
                "claim": claim,
                "value": None,
                "source_ids": [source_id],
                "source_quote": truncate(raw_text, max_quote_chars),
                "support_label": "supported",
            }
        )
    return evidence_units


def build_xbrl_evidence(
    facts: list[dict[str, Any]],
    max_quote_chars: int,
) -> list[dict[str, Any]]:
    evidence_units = []
    for idx, fact in enumerate(facts, start=1):
        source_id = str(fact.get("source_id", f"X{idx}"))
        claim = build_claim_from_fact(fact)
        evidence_units.append(
            {
                "evidence_id": f"E_XBRL_{idx:03d}",
                "category": infer_category_from_tag(fact.get("tag")),
                "claim": claim,
                "value": fact.get("value"),
                "source_ids": [source_id],
                "source_quote": truncate(claim, max_quote_chars),
                "support_label": "supported",
            }
        )
    return evidence_units


def build_price_context_evidence(
    price_context: dict[str, Any],
    max_quote_chars: int,
) -> list[dict[str, Any]]:
    available = []
    for key in ["ret_5d", "ret_20d", "market_ret_20d"]:
        value = price_context.get(key)
        if value is not None:
            available.append(f"{key}={value}")
    if not available:
        return []
    quote = "Pre-event price context: " + ", ".join(available)
    return [
        {
            "evidence_id": "E_CTX_001",
            "category": "pre_event_price_context",
            "claim": quote,
            "value": None,
            "source_ids": ["CTX1"],
            "source_quote": truncate(quote, max_quote_chars),
            "support_label": "supported",
        }
    ]


def build_bank(packet: dict[str, Any], max_quote_chars: int) -> dict[str, Any]:
    source_units = list(packet.get("source_units", []))
    xbrl_facts = list(packet.get("xbrl_facts", []))
    price_context = dict(packet.get("pre_event_price_context", {}))

    evidence_units = []
    evidence_units.extend(build_source_unit_evidence(source_units, max_quote_chars))
    evidence_units.extend(build_xbrl_evidence(xbrl_facts, max_quote_chars))
    evidence_units.extend(build_price_context_evidence(price_context, max_quote_chars))

    return {
        "event_id": packet["event_id"],
        "visibility_policy": packet.get("visibility_policy"),
        "identity_visibility": packet.get("identity_visibility"),
        "company_name": packet.get("company_name") or packet.get("company"),
        "ticker": packet.get("ticker"),
        "company_alias": packet.get("company_alias"),
        "masked_company": packet.get("masked_company"),
        "sector": packet.get("sector"),
        "market_cap_group": packet.get("market_cap_group"),
        "fiscal_period": packet.get("fiscal_period"),
        "event_date": packet.get("event_date"),
        "accepted_at_et": (
            packet.get("filing_metadata", {}).get("accepted_at_et")
            if isinstance(packet.get("filing_metadata"), dict)
            else None
        ),
        "source_packet_path": packet.get("_source_path"),
        "evidence_units": evidence_units,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_packets_dir = Path(args.source_packets_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    packets = load_source_packets(source_packets_dir)
    manifest_rows = []
    for packet in packets:
        bank = build_bank(packet, args.max_quote_chars)
        output_path = output_dir / f"{packet['event_id']}.json"
        write_json(output_path, bank)
        manifest_rows.append(
            {
                "event_id": packet["event_id"],
                "source_packet_path": packet.get("_source_path", ""),
                "evidence_bank_path": str(output_path),
                "source_unit_count": len(packet.get("source_units", [])),
                "xbrl_fact_count": len(packet.get("xbrl_facts", [])),
                "evidence_unit_count": len(bank["evidence_units"]),
            }
        )

    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, {"rows": manifest_rows})
    print(f"wrote {len(manifest_rows)} evidence banks to {output_dir}")
    print(f"wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
