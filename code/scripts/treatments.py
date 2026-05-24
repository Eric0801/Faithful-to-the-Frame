#!/usr/bin/env python3
"""Reference treatment renderer for the paper artifact release.

This script contains the public, compact treatment-generation logic used by the
paper. It avoids provider calls and operational batch machinery: T2/T3 are
emitted as prompt jobs, while T1/T4/B0 are deterministic renderings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


PROFILE_IDS = (
    "retail_day_trader",
    "retail_swing_trader",
    "retail_long_term_fundamental",
    "institutional_event_driven_hedge_fund",
    "institutional_prop_trader",
    "institutional_investment_advisor",
)
TREATMENT_T1 = "T1_raw_public_information"
TREATMENT_T2 = "T2_shared_summary"
TREATMENT_T3 = "T3_independent_summary"
TREATMENT_T4 = "T4_full_structured_evidence_ledger"
TREATMENT_B0 = "B0_canonical_evidence_only"
EVIDENCE_CATEGORY_ORDER = (
    "revenue",
    "earnings_or_eps",
    "margins",
    "guidance",
    "costs_or_expenses",
    "risk_or_uncertainty",
    "balance_sheet_or_cash",
    "pre_event_price_context",
    "other",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render")
    render.add_argument("--source-packets-dir", type=Path, default=None)
    render.add_argument("--evidence-banks-dir", type=Path, default=None)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--treatments", default="T1,T2,T3,T4,B0")
    render.add_argument("--profiles", default=",".join(PROFILE_IDS))
    render.add_argument("--representation-seeds", default="1,2")
    render.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def seed_items(value: str) -> list[int]:
    seeds = [int(item) for item in csv_items(value)]
    if not seeds:
        raise ValueError("at least one representation seed is required")
    return seeds


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def compact(value: Any) -> str:
    return " ".join(str(value if value is not None else "").split())


def load_event_json_dir(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    records: dict[str, dict[str, Any]] = {}
    for json_path in sorted(path.glob("*.json")):
        record = read_json(json_path)
        event_id = compact(record.get("event_id"))
        if not event_id:
            continue
        if event_id in records:
            raise ValueError(f"duplicate event_id: {event_id}")
        records[event_id] = record
    if not records:
        raise ValueError(f"no event JSON files found in {path}")
    return records


def source_id_sort_key(value: Any) -> tuple[str, int, str]:
    text = str(value)
    prefix = "".join(ch for ch in text if not ch.isdigit())
    digits = "".join(ch for ch in text if ch.isdigit())
    return (prefix, int(digits or 0), text)


def evidence_id_sort_key(value: Any) -> tuple[str, int, str]:
    return source_id_sort_key(value)


def company_label(record: dict[str, Any]) -> str:
    return (
        compact(record.get("company_name"))
        or compact(record.get("company"))
        or compact(record.get("company_alias"))
        or compact(record.get("masked_company"))
        or f"Company {compact(record.get('event_id')).split('_')[-1]}"
    )


def metadata_lines(record: dict[str, Any]) -> list[str]:
    fields = (
        ("Entity", company_label(record)),
        ("Ticker", record.get("ticker")),
        ("Event date", record.get("event_date")),
        ("Sector", record.get("sector")),
        ("Market cap group", record.get("market_cap_group")),
        ("Fiscal period", record.get("fiscal_period")),
    )
    return [f"{label}: {compact(value)}" for label, value in fields if compact(value)]


def render_t1_text(packet: dict[str, Any]) -> str:
    sections = ["\n".join(metadata_lines(packet))]
    source_units = packet.get("source_units") or []
    if source_units:
        lines = ["Source evidence:"]
        for unit in source_units:
            lines.append(
                f"- {compact(unit.get('source_id'))} "
                f"[{compact(unit.get('section')) or 'misc'}]: {compact(unit.get('text'))}"
            )
        sections.append("\n".join(lines))
    xbrl_facts = packet.get("xbrl_facts") or []
    if xbrl_facts:
        lines = ["Structured facts:"]
        for fact in xbrl_facts:
            value = compact(fact.get("value"))
            unit = compact(fact.get("unit"))
            lines.append(
                f"- {compact(fact.get('source_id'))} {compact(fact.get('tag'))}: "
                f"{value} {unit}".strip()
            )
        sections.append("\n".join(lines))
    price_context = packet.get("pre_event_price_context") or {}
    context_lines = [
        f"- {key}: {price_context[key]}"
        for key in ("ret_5d", "ret_20d", "market_ret_20d")
        if key in price_context and price_context[key] is not None
    ]
    if context_lines:
        sections.append("Pre-event price context:\n" + "\n".join(context_lines))
    return "\n\n".join(section for section in sections if section.strip())


def canonical_unit(unit: dict[str, Any]) -> dict[str, Any]:
    source_ids = sorted(
        [compact(item) for item in unit.get("source_ids", []) if compact(item)],
        key=source_id_sort_key,
    )
    return {
        "evidence_id": compact(unit.get("evidence_id")),
        "category": compact(unit.get("category")) or "other",
        "claim": compact(unit.get("claim")),
        "value": unit.get("value"),
        "source_ids": source_ids,
        "source_quote": compact(unit.get("source_quote")),
        "support_label": compact(unit.get("support_label")) or "supported",
    }


def canonical_units(bank: dict[str, Any]) -> list[dict[str, Any]]:
    units = [canonical_unit(unit) for unit in bank.get("evidence_units", [])]
    return sorted(units, key=lambda unit: evidence_id_sort_key(unit["evidence_id"]))


def category_order(category: str) -> tuple[int, str]:
    try:
        return (EVIDENCE_CATEGORY_ORDER.index(category), category)
    except ValueError:
        return (len(EVIDENCE_CATEGORY_ORDER), category)


def serialize_bank_for_prompt(bank: dict[str, Any]) -> str:
    fields = ("evidence_id", "category", "claim", "value", "source_ids", "support_label")
    rows = []
    for unit in canonical_units(bank):
        rows.append(json.dumps({field: unit.get(field) for field in fields}, sort_keys=True))
    return "\n".join(rows)


def build_summary_prompt(bank: dict[str, Any], treatment: str, seed: int) -> str:
    return (
        "You are rendering a source-grounded financial event representation.\n"
        f"Treatment: {treatment}\n"
        f"Representation seed: {seed}\n\n"
        "Rules:\n"
        "- Use only the supplied canonical evidence-unit bank.\n"
        "- Use source IDs for every factual claim.\n"
        "- Cite only source IDs in the form S### or X###.\n"
        "- Do not mention post-event outcomes, realized market reactions, hidden labels, future filings/news, or unsupported external facts.\n"
        "- Do not provide a buy/hold/sell recommendation.\n"
        "- Keep the output roughly 350-500 tokens.\n\n"
        "Event metadata:\n"
        + "\n".join(f"- {line}" for line in metadata_lines(bank))
        + "\n\nCanonical evidence-unit bank:\n"
        + serialize_bank_for_prompt(bank)
        + "\n"
    )


def derive_t3_seed(event_id: str, profile_id: str, representation_seed: int) -> int:
    token = f"{event_id}::{profile_id}::{representation_seed}"
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:12], 16)


def render_t4_text(bank: dict[str, Any]) -> str:
    units = canonical_units(bank)
    lines = [
        f"Entity: {company_label(bank)}",
        f"Event: {compact(bank.get('event_id'))}",
        "Representation: T4 full structured evidence ledger",
        "Design: deterministic lossless rendering of the full canonical evidence-unit bank.",
        "Rules: no evidence selection; no narrative synthesis; no action advice; no directional label.",
        f"Canonical evidence units included: {len(units)}",
    ]
    categories = sorted({unit["category"] for unit in units}, key=category_order)
    for category in categories:
        lines.append(f"## {category}")
        for unit in [item for item in units if item["category"] == category]:
            source_ids = ", ".join(unit["source_ids"])
            lines.extend(
                [
                    f"- evidence_id: {unit['evidence_id']}",
                    f"  category: {unit['category']}",
                    f"  claim: {unit['claim']}",
                    f"  value: {compact(unit['value'])}",
                    f"  source_ids: [{source_ids}]",
                    f"  support_label: {unit['support_label']}",
                    f"  source_quote: {unit['source_quote']}",
                ]
            )
    lines.append("No action advice or directional label is added by the renderer.")
    return "\n".join(lines)


def render_b0_text(bank: dict[str, Any]) -> str:
    return (
        "Event metadata:\n"
        + "\n".join(f"- {line}" for line in metadata_lines(bank))
        + "\n\nCanonical evidence-unit bank:\n"
        + serialize_bank_for_prompt(bank)
    )


def render_all(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    treatments = {item.upper() for item in csv_items(args.treatments)}
    seeds = seed_items(args.representation_seeds)
    profiles = csv_items(args.profiles)
    source_packets = load_event_json_dir(args.source_packets_dir) if "T1" in treatments else {}
    evidence_banks = (
        load_event_json_dir(args.evidence_banks_dir)
        if treatments.intersection({"T2", "T3", "T4", "B0"})
        else {}
    )

    if "T1" in treatments:
        for event_id, packet in source_packets.items():
            write_json(
                output_dir / TREATMENT_T1 / f"{event_id}.json",
                {
                    "event_id": event_id,
                    "treatment": TREATMENT_T1,
                    "representation_seed": 0,
                    "rendered_text": render_t1_text(packet),
                },
            )

    if "T2" in treatments:
        rows = [
            {
                "event_id": event_id,
                "treatment": TREATMENT_T2,
                "representation_seed": seed,
                "generator_seed": seed,
                "prompt": build_summary_prompt(bank, TREATMENT_T2, seed),
            }
            for event_id, bank in evidence_banks.items()
            for seed in seeds
        ]
        write_jsonl(output_dir / "prompt_jobs_T2_shared_summary.jsonl", rows)

    if "T3" in treatments:
        rows = [
            {
                "event_id": event_id,
                "treatment": TREATMENT_T3,
                "profile_id": profile,
                "representation_seed": seed,
                "generator_seed": derive_t3_seed(event_id, profile, seed),
                "prompt": build_summary_prompt(bank, TREATMENT_T3, seed),
            }
            for event_id, bank in evidence_banks.items()
            for seed in seeds
            for profile in profiles
        ]
        write_jsonl(output_dir / "prompt_jobs_T3_independent_summary.jsonl", rows)

    if "T4" in treatments:
        for event_id, bank in evidence_banks.items():
            units = canonical_units(bank)
            for seed in seeds:
                write_json(
                    output_dir / TREATMENT_T4 / f"{event_id}_rs{seed}.json",
                    {
                        "event_id": event_id,
                        "treatment": TREATMENT_T4,
                        "representation_seed": seed,
                        "rendered_text": render_t4_text(bank),
                        "canonical_evidence_unit_count": len(units),
                        "structured_evidence_units": units,
                    },
                )

    if "B0" in treatments:
        for event_id, bank in evidence_banks.items():
            write_json(
                output_dir / TREATMENT_B0 / f"{event_id}_rs0.json",
                {
                    "event_id": event_id,
                    "treatment": TREATMENT_B0,
                    "representation_seed": 0,
                    "rendered_text": render_b0_text(bank),
                    "canonical_evidence_unit_count": len(canonical_units(bank)),
                },
            )


def main() -> int:
    args = parse_args()
    if args.command == "render":
        render_all(args)
        return 0
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
