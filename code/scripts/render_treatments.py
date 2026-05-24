#!/usr/bin/env python3
"""Render deterministic treatments and prompt jobs from source packets.

This script keeps the phase-1 interfaces stable before the final data
collection is locked:

- T1: render an agent-facing raw packet text view from masked source packets
- T2: emit shared-summary prompt jobs from canonical evidence-unit banks
- T3: emit independent-summary prompt jobs from canonical evidence-unit banks
- T4: render deterministic full structured evidence ledgers from the same banks

The script does not call any model APIs. For T2/T3 it writes prompt bundles
that can later be executed by a model runner with full logging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "treatments"

DEFAULT_PROFILE_IDS = [
    "retail_day_trader",
    "retail_swing_trader",
    "retail_long_term_fundamental",
    "institutional_event_driven_hedge_fund",
    "institutional_prop_trader",
    "institutional_investment_advisor",
]

DEFAULT_REPRESENTATION_SEEDS = [1, 2]
DEFAULT_TREATMENTS = ["T1", "T2", "T3", "T4"]
T4_TREATMENT_ID = "T4_full_structured_evidence_ledger"

EVIDENCE_CATEGORY_ORDER = [
    "revenue",
    "earnings_or_eps",
    "margins",
    "guidance",
    "costs_or_expenses",
    "risk_or_uncertainty",
    "balance_sheet_or_cash",
    "pre_event_price_context",
]



@dataclass(frozen=True)
class RenderConfig:
    source_packets_dir: Path | None
    evidence_banks_dir: Path | None
    output_dir: Path
    treatments: list[str]
    profiles: list[str]
    representation_seeds: list[int]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seed_list(value: str) -> list[int]:
    seeds = []
    for item in parse_csv_list(value):
        try:
            seeds.append(int(item))
        except ValueError as exc:
            raise ValueError(f"invalid seed value: {item}") from exc
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-packets-dir", default="")
    parser.add_argument("--evidence-banks-dir", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--treatments",
        default=",".join(DEFAULT_TREATMENTS),
        help="Comma-separated subset of T1,T2,T3,T4",
    )
    parser.add_argument(
        "--profiles",
        default=",".join(DEFAULT_PROFILE_IDS),
        help="Comma-separated profile IDs for T3 prompt job expansion",
    )
    parser.add_argument(
        "--representation-seeds",
        default=",".join(str(seed) for seed in DEFAULT_REPRESENTATION_SEEDS),
        help="Comma-separated integer representation seeds",
    )
    return parser.parse_args()


def normalize_treatments(items: list[str]) -> list[str]:
    allowed = {"T1", "T2", "T3", "T4"}
    normalized = [item.upper() for item in items]
    invalid = [item for item in normalized if item not in allowed]
    if invalid:
        raise ValueError(f"unsupported treatments: {', '.join(invalid)}")
    return normalized


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json_dir(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for json_path in sorted(path.glob("*.json")):
        record = json.loads(json_path.read_text(encoding="utf-8"))
        event_id = str(record.get("event_id", "")).strip()
        if not event_id:
            continue
        if event_id in records:
            raise ValueError(f"duplicate event_id {event_id} in {path}")
        records[event_id] = record
    if not records:
        raise ValueError(f"no event JSON records found in {path}")
    return records


def derive_company_label(event_id: str) -> str:
    return f"Company {event_id.split('_')[-1]}"


def company_label(record: dict[str, Any]) -> str:
    return (
        record.get("company_name")
        or record.get("company")
        or record.get("company_alias")
        or record.get("masked_company")
        or derive_company_label(record["event_id"])
    )


def event_metadata_lines(record: dict[str, Any]) -> list[str]:
    lines = [f"Entity: {company_label(record)}"]
    ticker = record.get("ticker")
    if ticker:
        lines.append(f"Ticker: {ticker}")
    event_date = record.get("event_date")
    if event_date:
        lines.append(f"Event date: {event_date}")
    sector = record.get("sector")
    if sector:
        lines.append(f"Sector: {sector}")
    market_cap_group = record.get("market_cap_group")
    if market_cap_group:
        lines.append(f"Market cap group: {market_cap_group}")
    fiscal_period = record.get("fiscal_period")
    if fiscal_period:
        lines.append(f"Fiscal period: {fiscal_period}")
    return lines


def stringify_items(items: Any) -> str:
    if isinstance(items, list):
        return ", ".join(str(item) for item in items)
    if items is None:
        return ""
    return str(items)


def render_t1_text(packet: dict[str, Any]) -> str:
    filing_metadata = packet.get("filing_metadata", {})
    source_units = packet.get("source_units", [])
    xbrl_facts = packet.get("xbrl_facts", [])
    price_context = packet.get("pre_event_price_context", {})

    header_lines = event_metadata_lines(packet)
    accepted_at = filing_metadata.get("accepted_at_et")
    if accepted_at:
        header_lines.append(f"Accepted at ET: {accepted_at}")
    items_text = stringify_items(filing_metadata.get("items"))
    if items_text:
        header_lines.append(f"Items: {items_text}")

    sections = ["\n".join(header_lines)]
    if source_units:
        unit_lines = ["Source evidence:"]
        for unit in source_units:
            unit_lines.append(
                f"- {unit.get('source_id', 'S?')} [{unit.get('section', 'misc')}]: "
                f"{str(unit.get('text', '')).strip()}"
            )
        sections.append("\n".join(unit_lines))

    if xbrl_facts:
        fact_lines = ["Structured facts:"]
        for fact in xbrl_facts:
            value = fact.get("value")
            unit = fact.get("unit")
            fact_lines.append(
                f"- {fact.get('source_id', 'X?')} {fact.get('tag', 'unknown')}: "
                f"{value} {unit or ''}".strip()
            )
        sections.append("\n".join(fact_lines))

    if price_context:
        context_lines = ["Pre-event price context:"]
        for key in ["ret_5d", "ret_20d", "market_ret_20d"]:
            if key in price_context and price_context[key] is not None:
                context_lines.append(f"- {key}: {price_context[key]}")
        if len(context_lines) > 1:
            sections.append("\n".join(context_lines))

    return "\n\n".join(section for section in sections if section.strip())


def ordered_categories(units: list[dict[str, Any]]) -> list[str]:
    present = {str(unit.get("category", "other")) for unit in units}
    ordered = [name for name in EVIDENCE_CATEGORY_ORDER if name in present]
    extras = sorted(present.difference(EVIDENCE_CATEGORY_ORDER))
    return ordered + extras


def compact_whitespace(value: Any) -> str:
    return " ".join(str(value if value is not None else "").split())


def source_id_sort_key(source_id: Any) -> tuple[str, int, str]:
    text = str(source_id)
    prefix = "".join(ch for ch in text if not ch.isdigit())
    digits = "".join(ch for ch in text if ch.isdigit())
    return (prefix, int(digits or 0), text)


def evidence_id_sort_key(evidence_id: Any) -> tuple[str, int, str]:
    text = str(evidence_id)
    prefix = "".join(ch for ch in text if not ch.isdigit())
    digits = "".join(ch for ch in text if ch.isdigit())
    return (prefix, int(digits or 0), text)


def first_source_order(unit: dict[str, Any]) -> tuple[str, int, str]:
    source_ids = unit.get("source_ids") or []
    if not source_ids:
        return ("Z", 9999, "")
    return source_id_sort_key(source_ids[0])


def keyword_hits(text: str, keywords: list[str]) -> int:
    normalized = text.casefold()
    return sum(1 for keyword in keywords if keyword.casefold() in normalized)


def is_boilerplate_claim(text: str) -> bool:
    normalized = text.casefold()
    boilerplate_markers = [
        "/prnewswire/",
        "today reported",
        "president & ceo",
        "chief executive officer",
        "conference call",
        "investor relations",
    ]
    return any(marker in normalized for marker in boilerplate_markers)


def evidence_unit_sort_key(unit: dict[str, Any], keywords: list[str]) -> tuple[int, int, int, tuple[str, int, str], int]:
    claim = compact_whitespace(unit.get("claim"))
    hits = keyword_hits(claim, keywords)
    return (
        1 if is_boilerplate_claim(claim) else 0,
        0 if len(claim.split()) <= 45 else 1,
        -hits,
        first_source_order(unit),
        len(claim),
    )


def section_candidates(
    units: list[dict[str, Any]],
    categories: list[str],
    keywords: list[str],
    used_source_ids: set[str],
) -> list[dict[str, Any]]:
    category_set = set(categories)
    candidates = []
    for unit in units:
        if str(unit.get("category", "other")) not in category_set:
            continue
        source_ids = [str(item) for item in unit.get("source_ids", []) if str(item).strip()]
        if not source_ids:
            continue
        if used_source_ids.intersection(source_ids):
            continue
        candidates.append(unit)
    return sorted(candidates, key=lambda unit: evidence_unit_sort_key(unit, keywords))


def canonical_t4_unit(unit: dict[str, Any]) -> dict[str, Any]:
    source_ids = sorted(
        [str(item) for item in unit.get("source_ids", []) if str(item).strip()],
        key=source_id_sort_key,
    )
    return {
        "evidence_id": compact_whitespace(unit.get("evidence_id")),
        "category": compact_whitespace(unit.get("category") or "other"),
        "claim": compact_whitespace(unit.get("claim")),
        "value": unit.get("value"),
        "source_ids": source_ids,
        "source_quote": compact_whitespace(unit.get("source_quote")),
        "support_label": compact_whitespace(unit.get("support_label") or "supported"),
    }


def sorted_t4_units(bank: dict[str, Any]) -> list[dict[str, Any]]:
    units = [canonical_t4_unit(unit) for unit in bank.get("evidence_units", [])]
    return sorted(units, key=lambda unit: evidence_id_sort_key(unit["evidence_id"]))


def render_t4_unit(unit: dict[str, Any]) -> list[str]:
    source_ids = ", ".join(unit["source_ids"])
    lines = [
        f"- evidence_id: {unit['evidence_id']}",
        f"  category: {unit['category']}",
        f"  claim: {unit['claim']}",
    ]
    if unit["value"] is not None and compact_whitespace(unit["value"]):
        lines.append(f"  value: {compact_whitespace(unit['value'])}")
    lines.extend(
        [
            f"  source_ids: [{source_ids}]",
            f"  support_label: {unit['support_label']}",
            f"  source_quote: {unit['source_quote']}",
        ]
    )
    return lines


def render_t4_text(bank: dict[str, Any]) -> str:
    units = sorted_t4_units(bank)
    lines = [
        f"Entity: {company_label(bank)}",
        f"Event: {bank['event_id']}",
        "Representation: T4 full structured evidence ledger",
        "Design: deterministic lossless rendering of the full canonical evidence-unit bank.",
        "Rules: no evidence selection; no narrative synthesis; no action advice; no directional label.",
        f"Canonical evidence units included: {len(units)}",
    ]

    for category in ordered_categories(units):
        lines.append(f"## {category}")
        for unit in [item for item in units if item["category"] == category]:
            lines.extend(render_t4_unit(unit))

    lines.append("No action advice or directional label is added by the renderer.")
    return "\n".join(lines)


def serialize_evidence_bank(bank: dict[str, Any]) -> str:
    lines = []
    for unit in bank.get("evidence_units", []):
        entry = {
            "evidence_id": unit.get("evidence_id"),
            "category": unit.get("category"),
            "claim": unit.get("claim"),
            "value": unit.get("value"),
            "source_ids": unit.get("source_ids"),
            "source_quote": unit.get("source_quote"),
            "support_label": unit.get("support_label"),
        }
        lines.append(json.dumps(entry, ensure_ascii=True, sort_keys=True))
    return "\n".join(lines)


def serialize_event_metadata(bank: dict[str, Any]) -> str:
    return "\n".join(f"- {line}" for line in event_metadata_lines(bank))


def build_summary_prompt(
    bank: dict[str, Any],
    treatment: str,
    representation_seed: int,
) -> str:
    return (
        "You are rendering a source-grounded financial event representation.\n"
        f"Treatment: {treatment}\n"
        f"Representation seed: {representation_seed}\n\n"
        "Rules:\n"
        "- Use only the supplied canonical evidence-unit bank.\n"
        "- Use source IDs for every factual claim.\n"
        "- Cite only source IDs in the form S### or X###.\n"
        "- Never cite evidence IDs such as E_SRC_###, E_XBRL_###, or E_CTX_###.\n"
        "- Company, ticker, and event date may be mentioned only when supplied below.\n"
        "- Do not mention post-event outcomes, realized market reactions, hidden labels, future filings/news, or unsupported external facts.\n"
        "- Do not provide a buy/hold/sell recommendation.\n"
        "- Do not infer unsupported facts.\n"
        "- Keep the output roughly 350-500 tokens.\n\n"
        "Event metadata:\n"
        f"{serialize_event_metadata(bank)}\n\n"
        "Canonical evidence-unit bank:\n"
        f"{serialize_evidence_bank(bank)}\n"
    )


def derive_t3_generator_seed(event_id: str, profile_id: str, representation_seed: int) -> int:
    token = f"{event_id}::{profile_id}::{representation_seed}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            fh.write("\n")


def render_t1_packets(config: RenderConfig, source_packets: dict[str, dict[str, Any]]) -> int:
    output_dir = config.output_dir / "T1_raw_public_information"
    ensure_dir(output_dir)
    count = 0
    for event_id, packet in source_packets.items():
        payload = {
            "event_id": event_id,
            "treatment": "T1_raw_public_information",
            "representation_seed": 0,
            "rendered_text": render_t1_text(packet),
            "source_packet_path": packet.get("_source_path"),
        }
        write_json(output_dir / f"{event_id}.json", payload)
        count += 1
    return count


def render_t4_cards(config: RenderConfig, evidence_banks: dict[str, dict[str, Any]]) -> int:
    output_dir = config.output_dir / T4_TREATMENT_ID
    ensure_dir(output_dir)
    count = 0
    for event_id, bank in evidence_banks.items():
        for representation_seed in config.representation_seeds:
            payload = {
                "event_id": event_id,
                "treatment": T4_TREATMENT_ID,
                "representation_seed": representation_seed,
                "rendered_text": render_t4_text(bank),
                "evidence_bank_path": bank.get("_source_path"),
                "canonical_evidence_unit_count": len(bank.get("evidence_units", [])),
                "structured_evidence_units": sorted_t4_units(bank),
            }
            file_name = f"{event_id}_rs{representation_seed}.json"
            write_json(output_dir / file_name, payload)
            count += 1
    return count


def build_t2_jobs(config: RenderConfig, evidence_banks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for event_id, bank in evidence_banks.items():
        for representation_seed in config.representation_seeds:
            rows.append(
                {
                    "event_id": event_id,
                    "treatment": "T2_shared_summary",
                    "representation_seed": representation_seed,
                    "generator_seed": representation_seed,
                    "prompt": build_summary_prompt(bank, "T2_shared_summary", representation_seed),
                    "evidence_bank_path": bank.get("_source_path"),
                }
            )
    return rows


def build_t3_jobs(config: RenderConfig, evidence_banks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for event_id, bank in evidence_banks.items():
        for representation_seed in config.representation_seeds:
            for profile_id in config.profiles:
                rows.append(
                    {
                        "event_id": event_id,
                        "treatment": "T3_independent_summary",
                        "representation_seed": representation_seed,
                        "profile_id": profile_id,
                        "generator_seed": derive_t3_generator_seed(
                            event_id, profile_id, representation_seed
                        ),
                        "prompt": build_summary_prompt(
                            bank,
                            "T3_independent_summary",
                            representation_seed,
                        ),
                        "evidence_bank_path": bank.get("_source_path"),
                    }
                )
    return rows


def attach_source_paths(records: dict[str, dict[str, Any]], base_dir: Path) -> dict[str, dict[str, Any]]:
    updated = {}
    for json_path in sorted(base_dir.glob("*.json")):
        record = json.loads(json_path.read_text(encoding="utf-8"))
        event_id = str(record.get("event_id", "")).strip()
        if event_id in records:
            record["_source_path"] = str(json_path)
            updated[event_id] = record
    return updated


def validate_inputs(config: RenderConfig) -> None:
    if any(item in config.treatments for item in ["T1"]) and not config.source_packets_dir:
        raise ValueError("T1 requires --source-packets-dir")
    if any(item in config.treatments for item in ["T2", "T3", "T4"]) and not config.evidence_banks_dir:
        raise ValueError("T2/T3/T4 require --evidence-banks-dir")


def main() -> int:
    args = parse_args()
    config = RenderConfig(
        source_packets_dir=Path(args.source_packets_dir) if args.source_packets_dir else None,
        evidence_banks_dir=Path(args.evidence_banks_dir) if args.evidence_banks_dir else None,
        output_dir=Path(args.output_dir),
        treatments=normalize_treatments(parse_csv_list(args.treatments)),
        profiles=parse_csv_list(args.profiles),
        representation_seeds=parse_seed_list(args.representation_seeds),
    )
    validate_inputs(config)
    ensure_dir(config.output_dir)

    source_packets: dict[str, dict[str, Any]] = {}
    evidence_banks: dict[str, dict[str, Any]] = {}
    if config.source_packets_dir:
        source_packets = attach_source_paths(
            load_json_dir(config.source_packets_dir),
            config.source_packets_dir,
        )
    if config.evidence_banks_dir:
        evidence_banks = attach_source_paths(
            load_json_dir(config.evidence_banks_dir),
            config.evidence_banks_dir,
        )

    rendered_counts: dict[str, int] = {}
    if "T1" in config.treatments:
        rendered_counts["T1"] = render_t1_packets(config, source_packets)
    if "T4" in config.treatments:
        rendered_counts["T4"] = render_t4_cards(config, evidence_banks)
    if "T2" in config.treatments:
        rows = build_t2_jobs(config, evidence_banks)
        write_jsonl(config.output_dir / "prompt_jobs_T2_shared_summary.jsonl", rows)
        rendered_counts["T2"] = len(rows)
    if "T3" in config.treatments:
        rows = build_t3_jobs(config, evidence_banks)
        write_jsonl(config.output_dir / "prompt_jobs_T3_independent_summary.jsonl", rows)
        rendered_counts["T3"] = len(rows)

    print("render summary:")
    for key in sorted(rendered_counts):
        print(f"  {key}: {rendered_counts[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
