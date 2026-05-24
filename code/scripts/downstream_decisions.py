#!/usr/bin/env python3
"""Reference downstream decision request builder and output normalizer.

This script covers the experimental layer between rendered treatments and E2A
metric computation. It does not call model APIs. It writes downstream prompt
jobs and converts JSON model responses into the validated decision-row contract
consumed by ``metrics.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


PROFILE_DEFINITIONS = {
    "retail_day_trader": {
        "profile_group": "retail",
        "horizon": "1-5 trading days",
        "focus": ["short-term price reaction", "volume", "near-term catalyst"],
        "style": "tactical and risk-aware",
    },
    "retail_swing_trader": {
        "profile_group": "retail",
        "horizon": "several days to several weeks",
        "focus": ["trend continuation", "earnings surprise", "risk/reward"],
        "style": "balanced technical-fundamental",
    },
    "retail_long_term_fundamental": {
        "profile_group": "retail",
        "horizon": "multi-quarter",
        "focus": ["business fundamentals", "guidance", "valuation implications"],
        "style": "patient fundamental",
    },
    "institutional_event_driven_hedge_fund": {
        "profile_group": "institutional",
        "horizon": "event window",
        "focus": ["surprise versus expectations", "guidance", "positioning risk"],
        "style": "event-driven",
    },
    "institutional_prop_trader": {
        "profile_group": "institutional",
        "horizon": "intraday to several days",
        "focus": ["liquidity", "volatility", "near-term asymmetric setups"],
        "style": "fast tactical",
    },
    "institutional_investment_advisor": {
        "profile_group": "institutional",
        "horizon": "portfolio review cycle",
        "focus": ["client suitability", "fundamental risk", "portfolio fit"],
        "style": "risk-controlled advisory",
    },
}
ACTION_CODES = {"buy": 1, "hold": 0, "sell": -1}
DECISION_FIELDS = [
    "event_id",
    "treatment",
    "treatment_family",
    "upstream_model_family",
    "profile",
    "profile_group",
    "model_family",
    "representation_seed",
    "decision_seed",
    "expected_return_5d",
    "confidence",
    "action",
    "action_strength",
    "key_reasons",
    "evidence_used",
    "evidence_categories",
    "uncertainty_notes",
    "rationale",
]
DOWNSTREAM_INSTRUCTIONS = (
    "You are a downstream decision agent in a controlled experiment.\n\n"
    "Read only the supplied treatment packet and investor profile. Do not use "
    "external knowledge, browsing, post-event outcomes, realized price reactions, "
    "future filings/news, or hidden labels.\n\n"
    "Return strict JSON with keys: expected_return_5d, confidence, action, "
    "action_strength, key_reasons, evidence_used, evidence_categories, "
    "uncertainty_notes, rationale.\n"
    "- expected_return_5d must be a decimal return in [-0.20, 0.20].\n"
    "- confidence must be in [0, 1].\n"
    "- action must be one of buy, hold, sell.\n"
    "- action_strength must be in [-1, 1].\n"
    "- key_reasons must contain 2 to 5 concise reasons.\n"
    "- evidence_used should cite source IDs from the packet when available.\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-requests")
    build.add_argument("--rendered-treatments-dir", type=Path, required=True)
    build.add_argument("--output-path", type=Path, required=True)
    build.add_argument("--model-families", default="gpt-5.2")
    build.add_argument("--profiles", default=",".join(PROFILE_DEFINITIONS))
    build.add_argument("--decision-seeds", default="1,2")
    build.add_argument("--upstream-model-family", default="deterministic_or_released")

    normalize = subparsers.add_parser("normalize-outputs")
    normalize.add_argument("--provider-outputs-jsonl", type=Path, required=True)
    normalize.add_argument("--requests-jsonl", type=Path, default=None)
    normalize.add_argument("--output-csv", type=Path, required=True)
    normalize.add_argument("--output-jsonl", type=Path, default=None)
    normalize.add_argument("--strict", action="store_true")
    return parser.parse_args()


def csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def int_items(value: str) -> list[int]:
    return [int(item) for item in csv_items(value)]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(payload)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=DECISION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(serialize_decision_row(row))


def compact(value: Any) -> str:
    return " ".join(str(value if value is not None else "").split())


def treatment_family(treatment: str) -> str:
    text = treatment.strip()
    if text.startswith("B0"):
        return "B0"
    match = re.match(r"^(T\d)", text)
    return match.group(1) if match else text


def load_rendered_treatment_rows(path: Path) -> list[dict[str, Any]]:
    candidates = [path] if path.is_file() else sorted(path.rglob("*"))
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if not candidate.is_file() or candidate.suffix.lower() not in {".json", ".jsonl"}:
            continue
        if candidate.suffix.lower() == ".jsonl":
            for row in read_jsonl(candidate):
                if "rendered_text" in row:
                    row["_source_path"] = candidate.as_posix()
                    rows.append(row)
        else:
            row = read_json(candidate)
            if isinstance(row, dict) and "rendered_text" in row:
                row["_source_path"] = candidate.as_posix()
                rows.append(row)
    if not rows:
        raise ValueError(f"no rendered treatment rows found in {path}")
    return rows


def build_request_key(
    *,
    event_id: str,
    treatment: str,
    profile: str,
    model_family: str,
    representation_seed: int,
    decision_seed: int,
    upstream_model_family: str,
) -> str:
    return "|".join(
        [
            event_id,
            treatment,
            upstream_model_family,
            profile,
            model_family,
            str(representation_seed),
            str(decision_seed),
        ]
    )


def build_prompt(rendered_text: str, profile_id: str, profile: dict[str, Any], seed: int) -> str:
    focus = ", ".join(profile["focus"])
    return (
        DOWNSTREAM_INSTRUCTIONS
        + "\nTreatment packet:\n"
        + rendered_text.strip()
        + "\n\nInvestor profile:\n"
        + f"- profile: {profile_id}\n"
        + f"- profile_group: {profile['profile_group']}\n"
        + f"- horizon: {profile['horizon']}\n"
        + f"- focus: {focus}\n"
        + f"- style: {profile['style']}\n"
        + f"- decision_seed: {seed}\n"
    )


def build_requests(args: argparse.Namespace) -> int:
    profile_ids = csv_items(args.profiles)
    unknown = [profile_id for profile_id in profile_ids if profile_id not in PROFILE_DEFINITIONS]
    if unknown:
        raise ValueError(f"unknown profiles: {', '.join(unknown)}")
    model_families = csv_items(args.model_families)
    decision_seeds = int_items(args.decision_seeds)
    rows: list[dict[str, Any]] = []
    for treatment_row in load_rendered_treatment_rows(args.rendered_treatments_dir):
        event_id = compact(treatment_row.get("event_id"))
        treatment = compact(treatment_row.get("treatment"))
        rendered_text = compact(treatment_row.get("rendered_text"))
        if not event_id or not treatment or not rendered_text:
            continue
        representation_seed = int(treatment_row.get("representation_seed", 0))
        upstream_model_family = compact(treatment_row.get("upstream_model_family")) or args.upstream_model_family
        locked_profile = compact(treatment_row.get("profile_id"))
        row_profiles = [locked_profile] if locked_profile else profile_ids
        for model_family in model_families:
            for profile_id in row_profiles:
                profile = PROFILE_DEFINITIONS[profile_id]
                for decision_seed in decision_seeds:
                    request_key = build_request_key(
                        event_id=event_id,
                        treatment=treatment,
                        upstream_model_family=upstream_model_family,
                        profile=profile_id,
                        model_family=model_family,
                        representation_seed=representation_seed,
                        decision_seed=decision_seed,
                    )
                    rows.append(
                        {
                            "request_key": request_key,
                            "event_id": event_id,
                            "treatment": treatment,
                            "treatment_family": treatment_family(treatment),
                            "upstream_model_family": upstream_model_family,
                            "profile": profile_id,
                            "profile_group": profile["profile_group"],
                            "model_family": model_family,
                            "representation_seed": representation_seed,
                            "decision_seed": decision_seed,
                            "prompt": build_prompt(rendered_text, profile_id, profile, decision_seed),
                            "treatment_payload_path": treatment_row.get("_source_path", ""),
                        }
                    )
    write_jsonl(args.output_path, rows)
    print(f"wrote {len(rows)} downstream request rows to {args.output_path}")
    return 0


def maybe_parse_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def extract_decision_payload(row: dict[str, Any]) -> dict[str, Any]:
    for field in ("decision", "response_json", "parsed_response", "output"):
        payload = maybe_parse_json(row.get(field))
        if isinstance(payload, dict):
            return payload
    content = maybe_parse_json(row.get("content") or row.get("response") or row.get("text"))
    if isinstance(content, dict):
        return content
    return row


def request_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {row["request_key"]: row for row in read_jsonl(path) if row.get("request_key")}


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_list(value: Any) -> list[str]:
    value = maybe_parse_json(value)
    if isinstance(value, list):
        return [compact(item) for item in value if compact(item)]
    if value is None or value == "":
        return []
    return [item.strip() for item in re.split(r"[;,|]", str(value)) if item.strip()]


def validate_decision(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in (
        "event_id",
        "treatment",
        "profile",
        "profile_group",
        "model_family",
        "action",
    ):
        if not compact(row.get(field)):
            errors.append(f"missing {field}")
    expected = as_float(row.get("expected_return_5d"))
    confidence = as_float(row.get("confidence"))
    strength = as_float(row.get("action_strength"))
    if expected is None or not -0.20 <= expected <= 0.20:
        errors.append("expected_return_5d outside [-0.20, 0.20]")
    if confidence is None or not 0.0 <= confidence <= 1.0:
        errors.append("confidence outside [0, 1]")
    if strength is None or not -1.0 <= strength <= 1.0:
        errors.append("action_strength outside [-1, 1]")
    if compact(row.get("action")).lower() not in ACTION_CODES:
        errors.append("action must be buy, hold, or sell")
    if not as_list(row.get("key_reasons")):
        errors.append("key_reasons must be a non-empty list")
    return errors


def normalize_decision_row(
    provider_row: dict[str, Any],
    requests: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    request_key = compact(provider_row.get("request_key"))
    request = requests.get(request_key, {}) if request_key else {}
    payload = extract_decision_payload(provider_row)
    merged = {**request, **payload}
    if request_key:
        merged["request_key"] = request_key
    if "treatment_family" not in merged:
        merged["treatment_family"] = treatment_family(compact(merged.get("treatment")))
    merged["action"] = compact(merged.get("action")).lower()
    merged["expected_return_5d"] = as_float(merged.get("expected_return_5d"))
    merged["confidence"] = as_float(merged.get("confidence"))
    merged["action_strength"] = as_float(merged.get("action_strength"))
    merged["representation_seed"] = int(merged.get("representation_seed", 0))
    merged["decision_seed"] = int(merged.get("decision_seed", 0))
    for field in ("key_reasons", "evidence_used", "evidence_categories", "uncertainty_notes"):
        merged[field] = as_list(merged.get(field))
    if not compact(merged.get("rationale")):
        merged["rationale"] = " ".join(merged["key_reasons"])
    return merged, validate_decision(merged)


def serialize_decision_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    for field in ("key_reasons", "evidence_used", "evidence_categories", "uncertainty_notes"):
        output[field] = json.dumps(output.get(field, []), ensure_ascii=True)
    return output


def normalize_outputs(args: argparse.Namespace) -> int:
    requests = request_index(args.requests_jsonl)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for line_number, provider_row in enumerate(read_jsonl(args.provider_outputs_jsonl), start=1):
        normalized, errors = normalize_decision_row(provider_row, requests)
        if errors:
            failures.append({"line_number": line_number, "errors": errors})
        rows.append(normalized)
    if failures and args.strict:
        for failure in failures[:20]:
            print(f"ERROR line {failure['line_number']}: {failure['errors']}", file=sys.stderr)
        return 1
    write_csv(args.output_csv, rows)
    if args.output_jsonl is not None:
        write_jsonl(args.output_jsonl, [serialize_decision_row(row) for row in rows])
    print(f"wrote {len(rows)} decision rows to {args.output_csv}; validation_failures={len(failures)}")
    return 0 if not failures else 1


def main() -> int:
    args = parse_args()
    if args.command == "build-requests":
        return build_requests(args)
    if args.command == "normalize-outputs":
        return normalize_outputs(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
