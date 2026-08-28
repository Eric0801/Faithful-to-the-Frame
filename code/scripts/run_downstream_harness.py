#!/usr/bin/env python3
"""Execute downstream decision requests with resume-safe logging.

The harness consumes JSONL produced by ``build_downstream_requests.py`` and
writes decision-output JSONL rows that match the downstream validation schema.
Only a deterministic mock provider is implemented today, but the execution
path is separated behind a provider interface so real model backends can be
added later without changing the CLI contract.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TextIO

from harness_openai_compat import (
    DEFAULT_API_BASE_URL,
    OpenAICompatibleClient,
    OpenAICompatiblePermanentError,
    OpenAICompatibleTransientError,
    resolve_api_key,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "downstream_requests.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "decisions.jsonl"

REQUEST_KEY_FIELDS = (
    "event_id",
    "treatment",
    "upstream_model_family",
    "profile",
    "profile_group",
    "model_family",
    "representation_seed",
    "decision_seed",
)
REQUEST_REQUIRED_FIELDS = REQUEST_KEY_FIELDS + ("prompt",)
SOURCE_ID_PATTERN = re.compile(r"\b(?:S|X)\d{3}\b")
JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

PROFILE_BIASES = {
    "retail_day_trader": 0.018,
    "retail_swing_trader": 0.010,
    "retail_long_term_fundamental": 0.002,
    "institutional_event_driven_hedge_fund": 0.006,
    "institutional_prop_trader": -0.004,
    "institutional_investment_advisor": -0.010,
}
MODEL_BIASES = {
    "claude-sonnet-4.5": 0.006,
    "gpt-5.2": 0.000,
    "qwen3-235b-a22b": -0.004,
    "deepseek-v3.1": -0.002,
}
TREATMENT_BIASES = {
    "T1_raw_public_information": 0.000,
    "T1_full": 0.000,
    "T1_length_matched": -0.003,
    "T2_shared_summary": 0.005,
    "T2_shared_llm_summary": 0.005,
    "T3_independent_summary": 0.001,
    "T3_independent_summaries": 0.001,
    "T3_independent_llm_summary": 0.001,
    "T4_structured_evidence_card": -0.001,
    "T4_shared_atomic_evidence_view_control": -0.001,
    "T4_full_structured_evidence_ledger": -0.001,
    "T4_SAEV_deterministic": -0.001,
    "B0_canonical_evidence_only": 0.000,
    "T5_linguistic_deframing": 0.000,
    "T6_canonical_evidence_order_randomized": 0.000,
}
PROFILE_REASON_FOCUS = {
    "retail_day_trader": "short-horizon reaction and volatility",
    "retail_swing_trader": "post-earnings drift and guidance tone",
    "retail_long_term_fundamental": "durability of margins and balance-sheet quality",
    "institutional_event_driven_hedge_fund": "surprise magnitude and risk/reward symmetry",
    "institutional_prop_trader": "liquidity, velocity, and tactical asymmetry",
    "institutional_investment_advisor": "portfolio fit and drawdown discipline",
}
TREATMENT_REASON_NOTES = {
    "T1_raw_public_information": "The raw packet exposes direct source evidence without an added narrative layer.",
    "T1_full": "The raw packet exposes direct source evidence without an added narrative layer.",
    "T1_length_matched": "The packet stays close to raw evidence, so signal quality depends on direct source reading.",
    "T2_shared_summary": "The shared summary creates a cleaner but potentially more compressed signal mix.",
    "T2_shared_llm_summary": "The shared summary creates a cleaner but potentially more compressed signal mix.",
    "T3_independent_summary": "The independent summary preserves source grounding while allowing emphasis to vary.",
    "T3_independent_summaries": "The independent summary preserves source grounding while allowing emphasis to vary.",
    "T3_independent_llm_summary": "The independent summary preserves source grounding while allowing emphasis to vary.",
    "T4_structured_evidence_card": "The structured evidence card highlights cited facts without a narrative recommendation.",
    "T4_shared_atomic_evidence_view_control": "The shared atomic evidence view exposes bounded cited facts without a narrative recommendation.",
    "T4_full_structured_evidence_ledger": "The full structured evidence ledger exposes all canonical evidence units without narrative synthesis.",
    "T4_SAEV_deterministic": "The shared atomic evidence view exposes bounded cited facts without a narrative recommendation.",
    "B0_canonical_evidence_only": "The canonical evidence bank is supplied directly without upstream narrative synthesis.",
    "T5_linguistic_deframing": "The canonical evidence bank is supplied with pre-specified evaluative linguistic framing removed.",
    "T6_canonical_evidence_order_randomized": "The canonical evidence bank is supplied with a pre-specified randomized evidence-unit order.",
}
LEGACY_UPSTREAM_MODEL_FAMILY = "legacy_unspecified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--status-path",
        default=None,
        help="JSONL execution log path. Defaults next to the output file.",
    )
    parser.add_argument(
        "--provider",
        default="mock",
        help="Execution backend. Supported values: 'mock', 'openai-compatible'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect requests and print a summary without writing outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start a fresh output and status log instead of resuming.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Optional cap on requests to inspect or execute.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum provider attempts per request.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        default="0,1,2",
        help="Comma-separated backoff schedule for retries.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit after the first unrecoverable request failure.",
    )
    parser.add_argument(
        "--mock-fail-first-attempts",
        type=int,
        default=0,
        help="In mock mode, force the first N attempts of every request to fail.",
    )
    parser.add_argument(
        "--mock-latency-ms",
        type=int,
        default=0,
        help="Optional sleep per mock attempt to exercise timing logs.",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("OPENAI_API_BASE_URL", DEFAULT_API_BASE_URL),
        help="Base URL for an OpenAI-compatible API.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Explicit API key for openai-compatible mode. Prefer using an env var.",
    )
    parser.add_argument(
        "--api-key-env-var",
        default="OPENAI_API_KEY",
        help="Environment variable used to resolve the API key for openai-compatible mode.",
    )
    parser.add_argument(
        "--api-timeout-seconds",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds for openai-compatible mode.",
    )
    parser.add_argument(
        "--api-max-tokens",
        type=int,
        default=700,
        help="Suggested max_tokens for openai-compatible downstream decisions.",
    )
    parser.add_argument(
        "--api-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for openai-compatible downstream decisions.",
    )
    parser.add_argument(
        "--api-model-request-field",
        default="model_family",
        help="Request field used as the model name when --api-model-override is not set.",
    )
    parser.add_argument(
        "--api-model-override",
        default="",
        help="Optional model name override for openai-compatible mode.",
    )
    parser.add_argument(
        "--api-response-format-json",
        action="store_true",
        help="Request response_format={type:json_object} when the endpoint supports it.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent downstream requests to execute.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Deterministically partition requests into this many shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to execute when --shard-count > 1.",
    )
    return parser.parse_args()


def default_status_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.status{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.status.jsonl")


def parse_backoff_schedule(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("retry backoff schedule must contain at least one value")
    schedule = []
    for item in items:
        try:
            parsed = float(item)
        except ValueError as exc:
            raise ValueError(f"invalid retry backoff value: {item}") from exc
        if parsed < 0:
            raise ValueError(f"retry backoff must be non-negative: {item}")
        schedule.append(parsed)
    return schedule


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def shard_bucket(key: str, shard_count: int) -> int:
    return int(sha256_text(key)[:16], 16) % shard_count


def request_matches_shard(request_key: str, shard_count: int, shard_index: int) -> bool:
    if shard_count <= 1:
        return True
    return shard_bucket(request_key, shard_count) == shard_index


def json_candidates_from_text(value: str) -> list[str]:
    candidates: list[str] = []
    stripped = value.strip()
    if stripped:
        candidates.append(stripped)
    if stripped.startswith("```") and stripped.endswith("```"):
        inner = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        inner = re.sub(r"\s*```$", "", inner)
        if inner.strip():
            candidates.append(inner.strip())
    for match in JSON_FENCE_PATTERN.findall(value):
        if match.strip():
            candidates.append(match.strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1].strip())

    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def parse_json_object_from_text(value: str) -> dict[str, Any]:
    for candidate in json_candidates_from_text(value):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("provider response did not contain a valid JSON object")


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc.msg} column {exc.colno}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append((line_number, record))
    return rows


def is_t1_treatment(treatment: str) -> bool:
    return treatment in {
        "T1_raw_public_information",
        "T1_full",
        "T1_length_matched",
    } or treatment.startswith("T1_")


def default_upstream_model_family(record: dict[str, Any]) -> str:
    treatment = str(record.get("treatment", "")).strip()
    if is_t1_treatment(treatment):
        return "none"
    return LEGACY_UPSTREAM_MODEL_FAMILY


def normalize_request_record(record: dict[str, Any]) -> dict[str, Any]:
    upstream_model_family = str(record.get("upstream_model_family", "")).strip()
    if upstream_model_family:
        return record
    normalized = dict(record)
    normalized["upstream_model_family"] = default_upstream_model_family(record)
    return normalized


def resolve_reference_path(path_value: str, input_path: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate
    search_roots = (input_path.parent, PROJECT_ROOT, Path.cwd())
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (PROJECT_ROOT / candidate).resolve()


def build_request_key(record: dict[str, Any]) -> str:
    record = normalize_request_record(record)
    missing = [field for field in REQUEST_KEY_FIELDS if field not in record]
    if missing:
        raise ValueError(f"missing request key fields: {', '.join(missing)}")
    parts = []
    for field in REQUEST_KEY_FIELDS:
        value = record[field]
        parts.append(str(value).strip())
    return "|".join(parts)


def load_existing_request_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line_number, record in load_jsonl(path):
        try:
            request_key = build_request_key(record)
        except Exception as exc:
            raise ValueError(
                f"cannot resume from existing output {path}:{line_number}: {exc}"
            ) from exc
        if request_key in completed:
            raise ValueError(
                f"cannot resume from existing output {path}:{line_number}: "
                f"duplicate request key {request_key!r}. Deduplicate it or use "
                "--overwrite with a deliberate fresh run."
            )
        completed.add(request_key)
    return completed


def load_json_file(path: Path, cache: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    if path in cache:
        return cache[path]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    cache[path] = payload
    return payload


def load_jsonl_record(
    path: Path,
    line_number: int,
    cache: dict[tuple[Path, int], dict[str, Any]],
) -> dict[str, Any]:
    key = (path, line_number)
    if key in cache:
        return cache[key]
    if line_number < 1:
        raise ValueError(f"JSONL line number must be positive for {path}")
    with path.open("r", encoding="utf-8") as handle:
        for current_line, raw_line in enumerate(handle, start=1):
            if current_line != line_number:
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object in {path}:{line_number}")
            cache[key] = payload
            return payload
    raise ValueError(f"JSONL line {line_number} not found in {path}")


def split_jsonl_reference(path_value: str) -> tuple[str, int | None]:
    path_part, separator, line_part = path_value.rpartition(":")
    if separator and path_part and line_part.isdigit():
        return path_part, int(line_part)
    return path_value, None


def request_payload_line_number(request: dict[str, Any]) -> int | None:
    raw_value = request.get("treatment_payload_line_number")
    if raw_value is None or raw_value == "":
        return None
    if isinstance(raw_value, bool):
        raise ValueError("treatment_payload_line_number must be an integer")
    if isinstance(raw_value, int):
        if raw_value < 1:
            raise ValueError("treatment_payload_line_number must be positive")
        return raw_value
    if isinstance(raw_value, str) and raw_value.strip().isdigit():
        return int(raw_value.strip())
    raise ValueError("treatment_payload_line_number must be an integer")


def collect_source_ids_from_source_packet(packet: dict[str, Any]) -> list[str]:
    source_ids: list[str] = []
    for field in ("source_units", "xbrl_facts"):
        items = packet.get(field)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            source_id = item.get("source_id")
            if isinstance(source_id, str) and SOURCE_ID_PATTERN.fullmatch(source_id.strip()):
                source_ids.append(source_id.strip())
    return ordered_unique(source_ids)


def collect_source_ids_from_evidence_bank(bank: dict[str, Any]) -> list[str]:
    source_ids: list[str] = []
    units = bank.get("evidence_units")
    if not isinstance(units, list):
        return source_ids
    for unit in units:
        if not isinstance(unit, dict):
            continue
        values = unit.get("source_ids")
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and SOURCE_ID_PATTERN.fullmatch(value.strip()):
                source_ids.append(value.strip())
    return ordered_unique(source_ids)


def collect_source_ids_from_text(value: str) -> list[str]:
    return ordered_unique(SOURCE_ID_PATTERN.findall(value))


@dataclass(frozen=True)
class PayloadContext:
    payload_path: str | None
    payload_references: list[str]
    source_ids: list[str]
    warnings: list[str]


def extract_payload_context(
    request: dict[str, Any],
    input_path: Path,
    json_cache: dict[Path, dict[str, Any]],
    jsonl_cache: dict[tuple[Path, int], dict[str, Any]],
) -> PayloadContext:
    payload_path_value = request.get("treatment_payload_path")
    references: list[str] = []
    warnings: list[str] = []
    source_ids: list[str] = []
    payload_path_string: str | None = None

    if isinstance(payload_path_value, str) and payload_path_value.strip():
        raw_payload_path, inline_line_number = split_jsonl_reference(
            payload_path_value.strip()
        )
        line_number = request_payload_line_number(request) or inline_line_number
        payload_path = resolve_reference_path(raw_payload_path, input_path)
        payload_path_string = str(payload_path)
        if line_number is not None:
            payload_path_string = f"{payload_path}:{line_number}"
        if not payload_path.exists():
            warnings.append(f"treatment payload not found: {payload_path}")
        else:
            if line_number is not None:
                payload = load_jsonl_record(payload_path, line_number, jsonl_cache)
            else:
                payload = load_json_file(payload_path, json_cache)
            rendered_text = payload.get("rendered_text")
            if isinstance(rendered_text, str):
                source_ids.extend(collect_source_ids_from_text(rendered_text))

            source_packet_path = payload.get("source_packet_path")
            if isinstance(source_packet_path, str) and source_packet_path.strip():
                resolved_source_packet = resolve_reference_path(source_packet_path.strip(), payload_path)
                references.append(str(resolved_source_packet))
                if resolved_source_packet.exists():
                    source_packet = load_json_file(resolved_source_packet, json_cache)
                    source_ids.extend(collect_source_ids_from_source_packet(source_packet))
                else:
                    warnings.append(f"source packet not found: {resolved_source_packet}")

            evidence_bank_path = payload.get("evidence_bank_path")
            if isinstance(evidence_bank_path, str) and evidence_bank_path.strip():
                resolved_evidence_bank = resolve_reference_path(evidence_bank_path.strip(), payload_path)
                references.append(str(resolved_evidence_bank))
                if resolved_evidence_bank.exists():
                    evidence_bank = load_json_file(resolved_evidence_bank, json_cache)
                    source_ids.extend(collect_source_ids_from_evidence_bank(evidence_bank))
                    bank_source_packet_path = evidence_bank.get("source_packet_path")
                    if isinstance(bank_source_packet_path, str) and bank_source_packet_path.strip():
                        resolved_bank_source_packet = resolve_reference_path(
                            bank_source_packet_path.strip(),
                            resolved_evidence_bank,
                        )
                        references.append(str(resolved_bank_source_packet))
                        if resolved_bank_source_packet.exists():
                            source_packet = load_json_file(resolved_bank_source_packet, json_cache)
                            source_ids.extend(collect_source_ids_from_source_packet(source_packet))
                        else:
                            warnings.append(
                                f"evidence-bank source packet not found: {resolved_bank_source_packet}"
                            )
                else:
                    warnings.append(f"evidence bank not found: {resolved_evidence_bank}")
    else:
        warnings.append("request does not include treatment_payload_path")

    prompt = request.get("prompt")
    if isinstance(prompt, str):
        source_ids.extend(collect_source_ids_from_text(prompt))

    return PayloadContext(
        payload_path=payload_path_string,
        payload_references=ordered_unique(references),
        source_ids=ordered_unique(source_ids),
        warnings=warnings,
    )


@dataclass(frozen=True)
class PreparedRequest:
    line_number: int
    key: str
    request: dict[str, Any]
    prompt_sha256: str
    prompt_char_count: int
    payload: PayloadContext
    passthrough_metadata: dict[str, Any]


def prepare_request(
    line_number: int,
    record: dict[str, Any],
    input_path: Path,
    json_cache: dict[Path, dict[str, Any]],
    jsonl_cache: dict[tuple[Path, int], dict[str, Any]],
) -> PreparedRequest:
    record = normalize_request_record(record)
    missing = [field for field in REQUEST_REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"request line {line_number} missing fields: {', '.join(missing)}")

    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"request line {line_number} has empty prompt")

    key = build_request_key(record)
    payload = extract_payload_context(record, input_path, json_cache, jsonl_cache)
    passthrough_metadata = {
        key: value
        for key, value in record.items()
        if key not in REQUEST_KEY_FIELDS and key != "prompt"
    }
    return PreparedRequest(
        line_number=line_number,
        key=key,
        request=record,
        prompt_sha256=sha256_text(prompt),
        prompt_char_count=len(prompt),
        payload=payload,
        passthrough_metadata=passthrough_metadata,
    )


def pick_source_ids(source_ids: list[str], seed_material: str) -> list[str]:
    valid = [item for item in source_ids if SOURCE_ID_PATTERN.fullmatch(item)]
    if not valid:
        return ["S001"]
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    take_count = min(len(valid), 2 + (digest[0] % 2))
    start = digest[1] % len(valid)
    rotated = valid[start:] + valid[:start]
    return ordered_unique(rotated[:take_count])


def float_from_digest(digest: str, start: int) -> float:
    chunk = digest[start : start + 8]
    return int(chunk, 16) / 0xFFFFFFFF


def signed_from_digest(digest: str, start: int) -> float:
    return (2.0 * float_from_digest(digest, start)) - 1.0


def profile_focus(profile: str) -> str:
    return PROFILE_REASON_FOCUS.get(profile, "event-driven trade selection")


def treatment_note(treatment: str) -> str:
    return TREATMENT_REASON_NOTES.get(
        treatment,
        "The packet remains source-grounded, but signal emphasis depends on representation choices.",
    )


def build_action_from_score(score: float) -> tuple[str, float, float]:
    if abs(score) < 0.018:
        expected_return = round(clamp(score * 0.45, -0.20, 0.20), 4)
        action_strength = round(clamp(score * 6.0, -0.22, 0.22), 4)
        return "hold", expected_return, action_strength

    if score > 0:
        expected_return = round(clamp(abs(score) + 0.004, -0.20, 0.20), 4)
        action_strength = round(clamp((expected_return / 0.12), 0.20, 1.0), 4)
        return "buy", expected_return, action_strength

    expected_return = round(clamp(-abs(score) - 0.004, -0.20, 0.20), 4)
    action_strength = round(clamp((expected_return / 0.12), -1.0, -0.20), 4)
    return "sell", expected_return, action_strength


def build_key_reasons(
    action: str,
    prepared: PreparedRequest,
    evidence_used: list[str],
    expected_return: float,
) -> list[str]:
    profile = str(prepared.request["profile"])
    treatment = str(prepared.request["treatment"])
    evidence_text = ", ".join(evidence_used[:2])
    focus = profile_focus(profile)

    if action == "buy":
        signal_line = "The packet points to a modest positive skew rather than a neutral outcome."
    elif action == "sell":
        signal_line = "The packet points to a modest downside skew rather than a balanced setup."
    else:
        signal_line = "The packet mixes offsetting positives and negatives, which supports staying neutral."

    reasons = [
        signal_line,
        f"The {profile.replace('_', ' ')} lens emphasizes {focus}.",
        f"{treatment_note(treatment)} Key cited support comes from {evidence_text}.",
    ]

    if abs(expected_return) >= 0.04:
        reasons.append("The implied 5-day move is large enough to justify an active stance.")
    else:
        reasons.append("The implied 5-day move is small enough that sizing should stay controlled.")

    return reasons[:4]


def build_uncertainty_notes(action: str, prepared: PreparedRequest) -> list[str]:
    notes = [
        "Short-horizon reactions can diverge from fundamentals once the initial earnings narrative is digested.",
        "Signal strength is constrained by the packet only; no external expectations data is included.",
    ]
    if action == "hold":
        notes[0] = "The evidence mix is balanced enough that small interpretation changes could flip the trade direction."
    if not prepared.payload.source_ids:
        notes.append("Source ID recovery was incomplete, so evidence tracing relied on prompt text only.")
    return notes[:3]


@dataclass(frozen=True)
class ProviderResponse:
    decision_fields: dict[str, Any]
    raw_response: str
    provider_metadata: dict[str, Any] = field(default_factory=dict)


class DecisionProvider(Protocol):
    name: str

    def invoke(self, prepared: PreparedRequest, attempt: int) -> ProviderResponse:
        ...


class RetryableProviderError(RuntimeError):
    """Retryable provider execution failure."""


def provider_seed(prepared: PreparedRequest) -> int:
    digest = sha256_text(f"{prepared.key}|{prepared.prompt_sha256}")
    return int(digest[:12], 16)


@dataclass
class MockDecisionProvider:
    fail_first_attempts: int = 0
    latency_ms: int = 0
    name: str = "mock"

    def invoke(self, prepared: PreparedRequest, attempt: int) -> ProviderResponse:
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

        if attempt <= self.fail_first_attempts:
            raise RetryableProviderError(
                f"mock provider forced failure on attempt {attempt}"
            )

        request = prepared.request
        seed_material = f"{prepared.key}|{prepared.prompt_sha256}"
        digest = sha256_text(seed_material)

        base_score = (
            0.050 * signed_from_digest(digest, 0)
            + 0.018 * signed_from_digest(digest, 8)
            + PROFILE_BIASES.get(str(request["profile"]), 0.0)
            + MODEL_BIASES.get(str(request["model_family"]), 0.0)
            + TREATMENT_BIASES.get(str(request["treatment"]), 0.0)
            + (0.004 if int(request["decision_seed"]) == 1 else -0.004)
        )
        action, expected_return, action_strength = build_action_from_score(base_score)
        confidence = round(
            clamp(
                0.42
                + (0.32 * abs(action_strength))
                + (0.14 * float_from_digest(digest, 16))
                + (0.06 if action != "hold" else -0.02),
                0.0,
                1.0,
            ),
            4,
        )

        evidence_used = pick_source_ids(prepared.payload.source_ids, seed_material)
        decision_fields = {
            "expected_return_5d": expected_return,
            "confidence": confidence,
            "action": action,
            "action_strength": action_strength,
            "key_reasons": build_key_reasons(
                action,
                prepared,
                evidence_used,
                expected_return,
            ),
            "evidence_used": evidence_used,
            "uncertainty_notes": build_uncertainty_notes(action, prepared),
        }
        return ProviderResponse(
            decision_fields=decision_fields,
            raw_response=json.dumps(decision_fields, ensure_ascii=True, sort_keys=True),
            provider_metadata={
                "seed_material_sha256": digest,
                "mock_fail_first_attempts": self.fail_first_attempts,
            },
        )


class OpenAICompatibleDecisionProvider:
    """Decision provider backed by an OpenAI-compatible chat-completions API."""

    name = "openai-compatible"

    def __init__(
        self,
        *,
        api_base_url: str,
        api_key: str,
        api_key_env_var: str,
        timeout_seconds: float,
        max_tokens: int,
        temperature: float,
        model_request_field: str,
        model_override: str,
        response_format_json: bool,
    ) -> None:
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.api_key_env_var = api_key_env_var
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_request_field = model_request_field
        self.model_override = model_override.strip()
        self.response_format_json = response_format_json
        self._thread_local = threading.local()

    def _client_or_raise(self) -> OpenAICompatibleClient:
        client = getattr(self._thread_local, "client", None)
        if client is None:
            api_key = resolve_api_key(self.api_key, self.api_key_env_var)
            client = OpenAICompatibleClient(
                api_base_url=self.api_base_url,
                api_key=api_key,
                timeout_seconds=self.timeout_seconds,
            )
            self._thread_local.client = client
        return client

    def _model_for_request(self, prepared: PreparedRequest) -> str:
        if self.model_override:
            return self.model_override
        value = prepared.request.get(self.model_request_field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError(
            f"request missing model field '{self.model_request_field}' and no override was provided"
        )

    def invoke(self, prepared: PreparedRequest, attempt: int) -> ProviderResponse:
        model = self._model_for_request(prepared)
        response_format = {"type": "json_object"} if self.response_format_json else None
        try:
            result = self._client_or_raise().chat_completion(
                model=model,
                user_prompt=str(prepared.request["prompt"]),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=provider_seed(prepared),
                response_format=response_format,
            )
        except OpenAICompatibleTransientError as exc:
            raise RetryableProviderError(str(exc)) from exc
        except OpenAICompatiblePermanentError as exc:
            raise ValueError(str(exc)) from exc

        try:
            decision_fields = parse_json_object_from_text(result.text)
        except ValueError as exc:
            raise RetryableProviderError(str(exc)) from exc

        required = [
            "expected_return_5d",
            "confidence",
            "action",
            "action_strength",
        ]
        missing = [field for field in required if field not in decision_fields]
        if missing:
            raise RetryableProviderError(
                f"provider response JSON missing required fields: {', '.join(missing)}"
            )

        return ProviderResponse(
            decision_fields=decision_fields,
            raw_response=result.text,
            provider_metadata={
                "api_response_id": result.response_id,
                "api_response_model": result.response_model or model,
                "api_usage": result.usage,
                "api_model_request_field": self.model_request_field,
                "api_model_override": self.model_override or None,
                "api_response_format_json": self.response_format_json,
            },
        )


def build_provider(args: argparse.Namespace) -> DecisionProvider:
    provider_name = args.provider.strip().lower()
    if provider_name == "mock":
        return MockDecisionProvider(
            fail_first_attempts=max(0, args.mock_fail_first_attempts),
            latency_ms=max(0, args.mock_latency_ms),
        )
    if provider_name in {"openai-compatible", "openai_compatible", "openai"}:
        return OpenAICompatibleDecisionProvider(
            api_base_url=args.api_base_url,
            api_key=args.api_key,
            api_key_env_var=args.api_key_env_var,
            timeout_seconds=args.api_timeout_seconds,
            max_tokens=args.api_max_tokens,
            temperature=args.api_temperature,
            model_request_field=args.api_model_request_field,
            model_override=args.api_model_override,
            response_format_json=args.api_response_format_json,
        )
    raise ValueError(f"provider '{args.provider}' is not implemented yet")


def coerce_string_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        output = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if output:
            return output
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return fallback
        try:
            parsed = json.loads(stripped)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            output = [item.strip() for item in parsed if isinstance(item, str) and item.strip()]
            if output:
                return output
        lines = []
        for raw_line in stripped.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line[:1] in {"-", "*", "\u2022"}:
                line = line[1:].strip()
            if line:
                lines.append(line)
        if len(lines) > 1:
            return lines
        return [stripped]
    return fallback


def coerce_source_id_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        output = [
            item.strip()
            for item in value
            if isinstance(item, str) and SOURCE_ID_PATTERN.fullmatch(item.strip())
        ]
        if output:
            return ordered_unique(output)
    if isinstance(value, str):
        output = collect_source_ids_from_text(value)
        if output:
            return output
    return fallback


def build_output_record(
    prepared: PreparedRequest,
    provider: DecisionProvider,
    response: ProviderResponse,
    attempt: int,
    latency_ms: int,
) -> dict[str, Any]:
    request = prepared.request
    decision_fields = response.decision_fields
    evidence_used = coerce_source_id_list(
        decision_fields.get("evidence_used"),
        pick_source_ids(prepared.payload.source_ids, prepared.key),
    )
    key_reasons = coerce_string_list(
        decision_fields.get("key_reasons"),
        ["Synthetic mock output preserves the downstream schema.", "No external information was used."],
    )[:5]
    if len(key_reasons) < 2:
        key_reasons.append("The response is deterministic so it can be resumed and diffed safely.")
    uncertainty_notes = coerce_string_list(
        decision_fields.get("uncertainty_notes"),
        ["Synthetic mock response for harness validation."],
    )

    output = {
        field: request[field]
        for field in REQUEST_KEY_FIELDS
    }
    output.update(
        {
            "expected_return_5d": round(
                clamp(float(decision_fields["expected_return_5d"]), -0.20, 0.20),
                4,
            ),
            "confidence": round(
                clamp(float(decision_fields["confidence"]), 0.0, 1.0),
                4,
            ),
            "action": str(decision_fields["action"]).strip().lower(),
            "action_strength": round(
                clamp(float(decision_fields["action_strength"]), -1.0, 1.0),
                4,
            ),
            "key_reasons": key_reasons[:5],
            "evidence_used": evidence_used,
            "uncertainty_notes": uncertainty_notes,
            "request_key": prepared.key,
            "request_metadata": {
                "line_number": prepared.line_number,
                "prompt_sha256": prepared.prompt_sha256,
                "prompt_char_count": prepared.prompt_char_count,
                "treatment_payload_path": prepared.payload.payload_path,
                "payload_references": prepared.payload.payload_references,
                "payload_warnings": prepared.payload.warnings,
                "passthrough_fields": prepared.passthrough_metadata,
            },
            "execution_metadata": {
                "provider": provider.name,
                "attempt": attempt,
                "latency_ms": latency_ms,
                "completed_at_utc": utc_now_iso(),
                "raw_response": response.raw_response,
                "provider_metadata": response.provider_metadata,
            },
        }
    )
    return output


def append_jsonl(handle: TextIO, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    handle.write("\n")
    handle.flush()


def make_status_record(
    prepared: PreparedRequest,
    provider_name: str,
    status: str,
    *,
    attempt: int | None = None,
    latency_ms: int | None = None,
    message: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if message is None and error is not None:
        message = error
    record = {
        "timestamp_utc": utc_now_iso(),
        "status": status,
        "provider": provider_name,
        "request_key": prepared.key,
        "request_line_number": prepared.line_number,
        "event_id": prepared.request.get("event_id"),
        "treatment": prepared.request.get("treatment"),
        "upstream_model_family": prepared.request.get("upstream_model_family"),
        "profile": prepared.request.get("profile"),
        "model_family": prepared.request.get("model_family"),
        "representation_seed": prepared.request.get("representation_seed"),
        "decision_seed": prepared.request.get("decision_seed"),
    }
    if attempt is not None:
        record["attempt"] = attempt
    if latency_ms is not None:
        record["latency_ms"] = latency_ms
    if message is not None:
        record["message"] = message
    if error is not None:
        record["error"] = error
    return record


def log_status(
    handle: TextIO | None,
    prepared: PreparedRequest,
    provider_name: str,
    status: str,
    *,
    attempt: int | None = None,
    latency_ms: int | None = None,
    message: str | None = None,
    error: str | None = None,
) -> None:
    if handle is None:
        return
    append_jsonl(
        handle,
        make_status_record(
            prepared,
            provider_name,
            status,
            attempt=attempt,
            latency_ms=latency_ms,
            message=message,
            error=error,
        ),
    )


@dataclass
class RunSummary:
    total_requests_seen: int = 0
    total_requests_shard_selected: int = 0
    total_requests_selected: int = 0
    skipped_existing: int = 0
    succeeded: int = 0
    failed: int = 0


@dataclass(frozen=True)
class RequestExecutionOutcome:
    prepared: PreparedRequest
    succeeded: bool
    attempt_count: int
    output_record: dict[str, Any] | None
    status_records: tuple[dict[str, Any], ...]


def execute_request(
    prepared: PreparedRequest,
    provider: DecisionProvider,
    max_attempts: int,
    retry_backoff_seconds: list[float],
) -> RequestExecutionOutcome:
    status_records: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        status_records.append(
            make_status_record(
                prepared,
                provider.name,
                "attempt_started",
                attempt=attempt,
            )
        )
        start = time.perf_counter()
        try:
            response = provider.invoke(prepared, attempt)
        except RetryableProviderError as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            status_records.append(
                make_status_record(
                    prepared,
                    provider.name,
                    "attempt_failed",
                    attempt=attempt,
                    latency_ms=latency_ms,
                    error=str(exc),
                )
            )
            if attempt >= max_attempts:
                status_records.append(
                    make_status_record(
                        prepared,
                        provider.name,
                        "failed",
                        attempt=attempt,
                        latency_ms=latency_ms,
                        error=str(exc),
                    )
                )
                return RequestExecutionOutcome(
                    prepared=prepared,
                    succeeded=False,
                    attempt_count=attempt,
                    output_record=None,
                    status_records=tuple(status_records),
                )
            backoff = retry_backoff_seconds[min(attempt - 1, len(retry_backoff_seconds) - 1)]
            status_records.append(
                make_status_record(
                    prepared,
                    provider.name,
                    "retry_scheduled",
                    attempt=attempt,
                    message=f"sleeping {backoff:.3f}s before retry",
                )
            )
            if backoff > 0:
                time.sleep(backoff)
            continue
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            status_records.append(
                make_status_record(
                    prepared,
                    provider.name,
                    "failed",
                    attempt=attempt,
                    latency_ms=latency_ms,
                    error=str(exc),
                )
            )
            return RequestExecutionOutcome(
                prepared=prepared,
                succeeded=False,
                attempt_count=attempt,
                output_record=None,
                status_records=tuple(status_records),
            )

        latency_ms = int((time.perf_counter() - start) * 1000)
        output_record = build_output_record(prepared, provider, response, attempt, latency_ms)
        status_records.append(
            make_status_record(
                prepared,
                provider.name,
                "succeeded",
                attempt=attempt,
                latency_ms=latency_ms,
            )
        )
        return RequestExecutionOutcome(
            prepared=prepared,
            succeeded=True,
            attempt_count=attempt,
            output_record=output_record,
            status_records=tuple(status_records),
        )

    return RequestExecutionOutcome(
        prepared=prepared,
        succeeded=False,
        attempt_count=max_attempts,
        output_record=None,
        status_records=tuple(status_records),
    )


def print_dry_run_summary(
    input_path: Path,
    output_path: Path,
    provider: DecisionProvider,
    summary: RunSummary,
    shard_count: int,
    shard_index: int,
) -> None:
    print(f"dry_run=true input={input_path}")
    print(f"provider={provider.name} output={output_path}")
    print(
        f"requests_seen={summary.total_requests_seen} "
        f"requests_selected={summary.total_requests_selected}"
    )
    if shard_count > 1:
        print(
            f"requests_shard_selected={summary.total_requests_shard_selected} "
            f"shard={shard_index}/{shard_count}"
        )


def print_run_summary(
    input_path: Path,
    output_path: Path,
    status_path: Path,
    provider: DecisionProvider,
    summary: RunSummary,
    shard_count: int,
    shard_index: int,
) -> None:
    print(f"input={input_path}")
    print(f"provider={provider.name}")
    print(f"output={output_path}")
    print(f"status_log={status_path}")
    print(
        f"requests_seen={summary.total_requests_seen} "
        f"requests_selected={summary.total_requests_selected} "
        f"skipped_existing={summary.skipped_existing} "
        f"succeeded={summary.succeeded} failed={summary.failed}"
    )
    if shard_count > 1:
        print(
            f"requests_shard_selected={summary.total_requests_shard_selected} "
            f"shard={shard_index}/{shard_count}"
        )


def main() -> int:
    args = parse_args()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.max_requests is not None and args.max_requests < 0:
        raise ValueError("--max-requests must be non-negative")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count")
    if args.api_timeout_seconds <= 0:
        raise ValueError("--api-timeout-seconds must be > 0")
    if args.api_max_tokens <= 0:
        raise ValueError("--api-max-tokens must be > 0")
    if not 0.0 <= args.api_temperature <= 2.0:
        raise ValueError("--api-temperature must be between 0 and 2")

    input_path = Path(args.input)
    output_path = Path(args.output_path)
    status_path = Path(args.status_path) if args.status_path else default_status_path(output_path)
    retry_backoff_seconds = parse_backoff_schedule(args.retry_backoff_seconds)

    provider_name = args.provider.strip().lower()
    json_cache: dict[Path, dict[str, Any]] = {}
    jsonl_cache: dict[tuple[Path, int], dict[str, Any]] = {}
    all_request_rows = load_jsonl(input_path)
    all_prepared_requests = [
        prepare_request(line_number, record, input_path, json_cache, jsonl_cache)
        for line_number, record in all_request_rows
    ]
    shard_selected_requests = [
        prepared
        for prepared in all_prepared_requests
        if request_matches_shard(prepared.key, args.shard_count, args.shard_index)
    ]
    prepared_requests = shard_selected_requests
    if args.max_requests is not None:
        prepared_requests = prepared_requests[: args.max_requests]

    summary = RunSummary(
        total_requests_seen=len(all_request_rows),
        total_requests_shard_selected=len(shard_selected_requests),
        total_requests_selected=len(prepared_requests),
    )

    if args.dry_run:
        print_dry_run_summary(
            input_path,
            output_path,
            provider=type("DryRunProvider", (), {"name": provider_name})(),
            summary=summary,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )
        return 0

    provider = build_provider(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        output_path.write_text("", encoding="utf-8")
        status_path.write_text("", encoding="utf-8")

    completed_keys = load_existing_request_keys(output_path)

    with output_path.open("a", encoding="utf-8") as output_handle, status_path.open(
        "a",
        encoding="utf-8",
    ) as status_handle:
        scheduled_requests: list[PreparedRequest] = []
        reserved_keys = set(completed_keys)
        for prepared in prepared_requests:
            if prepared.key in reserved_keys:
                summary.skipped_existing += 1
                log_status(
                    status_handle,
                    prepared,
                    provider.name,
                    "skipped_existing",
                    message="output row already present or queued earlier in this run",
                )
                continue
            reserved_keys.add(prepared.key)
            scheduled_requests.append(prepared)

        if args.workers <= 1:
            for prepared in scheduled_requests:
                outcome = execute_request(
                    prepared,
                    provider,
                    args.max_attempts,
                    retry_backoff_seconds,
                )
                for record in outcome.status_records:
                    append_jsonl(status_handle, record)
                if outcome.succeeded:
                    assert outcome.output_record is not None
                    append_jsonl(output_handle, outcome.output_record)
                    completed_keys.add(prepared.key)
                    summary.succeeded += 1
                    continue
                summary.failed += 1
                if args.fail_fast:
                    print_run_summary(input_path, output_path, status_path, provider, summary)
                    return 1
        else:
            stop_after_failure = False
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                future_to_prepared = {
                    pool.submit(
                        execute_request,
                        prepared,
                        provider,
                        args.max_attempts,
                        retry_backoff_seconds,
                    ): prepared
                    for prepared in scheduled_requests
                }
                for future in as_completed(future_to_prepared):
                    prepared = future_to_prepared[future]
                    try:
                        outcome = future.result()
                    except Exception as exc:
                        outcome = RequestExecutionOutcome(
                            prepared=prepared,
                            succeeded=False,
                            attempt_count=1,
                            output_record=None,
                            status_records=(
                                make_status_record(
                                    prepared,
                                    provider.name,
                                    "failed",
                                    attempt=1,
                                    error=f"unexpected worker error: {exc}",
                                ),
                            ),
                        )
                    for record in outcome.status_records:
                        append_jsonl(status_handle, record)
                    if outcome.succeeded:
                        assert outcome.output_record is not None
                        append_jsonl(output_handle, outcome.output_record)
                        completed_keys.add(prepared.key)
                        summary.succeeded += 1
                        continue
                    summary.failed += 1
                    if args.fail_fast:
                        stop_after_failure = True
                        for pending_future in future_to_prepared:
                            if pending_future is future:
                                continue
                            pending_future.cancel()
                        break
            if stop_after_failure:
                print_run_summary(
                    input_path,
                    output_path,
                    status_path,
                    provider,
                    summary,
                    args.shard_count,
                    args.shard_index,
                )
                return 1

    print_run_summary(
        input_path,
        output_path,
        status_path,
        provider,
        summary,
        args.shard_count,
        args.shard_index,
    )
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"run_downstream_harness failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
