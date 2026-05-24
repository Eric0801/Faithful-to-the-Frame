#!/usr/bin/env python3
"""Run a resumable representation-generation harness over prompt-job JSONL files.

This harness consumes prompt-job JSONL files emitted by `render_treatments.py`
and writes rendered representation rows as JSONL. It currently supports:

- `dry-run` mode: validate inputs, apply resume logic, and emit per-row status logs
- `mock` mode: deterministically render source-grounded placeholder summaries

The execution interface is provider-agnostic so a real API-backed executor can
be added later without changing the CLI contract or output shape.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO

from harness_openai_compat import (
    DEFAULT_API_BASE_URL,
    OpenAICompatibleClient,
    OpenAICompatiblePermanentError,
    OpenAICompatibleTransientError,
    resolve_api_key,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "representations.jsonl"

PROMPT_BANK_MARKER = "Canonical evidence-unit bank:\n"
DEFAULT_MOCK_PROVIDER = "mock"
DEFAULT_MOCK_MODEL = "mock-neutral-analyst-v1"
LEGACY_UNKNOWN_UPSTREAM_MODEL_FAMILY = "legacy_unspecified"
SOURCE_ID_PATTERN = re.compile(r"\b(?:S|X)\d{3}\b")
EVIDENCE_ID_PATTERN = re.compile(r"\bE_(?:SRC|XBRL|CTX)_\d{3}\b")

CATEGORY_LABELS = {
    "revenue": "Revenue",
    "earnings_or_eps": "Earnings and EPS",
    "margins": "Margins",
    "guidance": "Guidance",
    "costs_or_expenses": "Costs and expenses",
    "risk_or_uncertainty": "Risk and uncertainty",
    "balance_sheet_or_cash": "Balance sheet and cash",
    "pre_event_price_context": "Pre-event price context",
}

INTRO_TEMPLATES = (
    "The canonical evidence bank describes a neutral operating update built from disclosed quantitative and qualitative items.",
    "The source-grounded bank presents a bounded event summary anchored in the supplied evidence units.",
    "The disclosed evidence outlines a mixed company update using only the canonical bank provided for this treatment.",
)

CATEGORY_TEMPLATES = (
    "{label} evidence notes {claims}.",
    "Within {label_lower}, the bank records {claims}.",
    "{label} detail includes {claims}.",
)

CLOSING_TEMPLATES = (
    "Taken together, the bank combines operating results, guidance, and risk context without implying a directional recommendation.",
    "Overall, the evidence remains source-grounded and descriptive rather than prescriptive.",
    "In aggregate, the bank summarizes disclosed fundamentals and management context without a trade call.",
)

EXACT_DATE_PATTERN = re.compile(
    r"\b("
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r")\s+\d{1,2}(?:,\s+\d{4})?\b"
)
TICKER_PATTERN = re.compile(r"\b(?:Nasdaq|NASDAQ|NYSE|NYSE American)\s*:\s*[^),;]+")
MASKED_COMPANY_PATTERN = re.compile(r"\bCompany\s+[A-Z](?:\s+Corporation)?\b")


class PromptJobError(ValueError):
    """Raised when a prompt-job row is malformed."""


class TransientExecutionError(RuntimeError):
    """Raised when an executor encounters a retryable error."""


class PermanentExecutionError(RuntimeError):
    """Raised when an executor encounters a non-retryable error."""


@dataclass(frozen=True)
class PromptJob:
    source_path: Path
    line_number: int
    event_id: str
    treatment: str
    representation_seed: int
    generator_seed: int
    prompt: str
    profile_id: str | None
    evidence_bank_path: str | None
    raw_record: dict[str, Any]
    job_key: str
    prompt_sha256: str


@dataclass(frozen=True)
class ExecutionConfig:
    mode: str
    provider: str
    model: str
    upstream_model_family: str
    max_retries: int
    retry_backoff_seconds: float
    mock_retry_mod: int
    api_base_url: str
    api_key: str
    api_key_env_var: str
    api_timeout_seconds: float
    api_max_tokens: int
    api_temperature: float
    workers: int


@dataclass(frozen=True)
class ExecutionRequest:
    job: PromptJob
    config: ExecutionConfig


@dataclass(frozen=True)
class ExecutionResult:
    rendered_text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetryNotice:
    attempt_number: int
    retry_count: int
    message: str


@dataclass(frozen=True)
class JobExecutionOutcome:
    job: PromptJob
    result: ExecutionResult | None
    attempts: int
    retry_notices: tuple[RetryNotice, ...]
    error_message: str | None

    @property
    def success(self) -> bool:
        return self.result is not None


@dataclass
class RunSummary:
    input_files: int = 0
    seen_rows: int = 0
    selected_rows: int = 0
    invalid_rows: int = 0
    skipped_existing: int = 0
    dry_run_rows: int = 0
    completed_rows: int = 0
    failed_rows: int = 0
    total_attempts: int = 0
    total_retries: int = 0
    retried_jobs: int = 0
    existing_output_rows: int = 0
    existing_output_invalid: int = 0
    existing_output_duplicates: int = 0


class RepresentationExecutor(ABC):
    """Provider-agnostic execution interface for representation generation."""

    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def execute(self, request: ExecutionRequest, attempt_number: int) -> ExecutionResult:
        raise NotImplementedError


class MockRepresentationExecutor(RepresentationExecutor):
    """Deterministic executor used for local harness testing."""

    @property
    def name(self) -> str:
        return "mock"

    def execute(self, request: ExecutionRequest, attempt_number: int) -> ExecutionResult:
        if self._should_inject_retry(request.job.job_key) and attempt_number == 1:
            raise TransientExecutionError("deterministic mock transient failure")

        rendered_text, derived_seed, unit_count, category_count = render_mock_representation(
            request.job
        )
        metadata = {
            "mock_template_version": "v1",
            "mock_seed": derived_seed,
            "mock_evidence_unit_count": unit_count,
            "mock_category_count": category_count,
            "prompt_char_count": len(request.job.prompt),
            "rendered_char_count": len(rendered_text),
            "rendered_word_count": len(rendered_text.split()),
        }
        return ExecutionResult(rendered_text=rendered_text, metadata=metadata)

    def _should_inject_retry(self, job_key: str) -> bool:
        mod = self.config.mock_retry_mod
        if mod <= 0:
            return False
        token = int(job_key[:12], 16)
        return token % mod == 0


class OpenAICompatibleRepresentationExecutor(RepresentationExecutor):
    """Real executor backed by an OpenAI-compatible chat-completions API."""

    def __init__(self, config: ExecutionConfig) -> None:
        super().__init__(config)
        self.api_key = resolve_api_key(config.api_key, config.api_key_env_var)
        self._thread_local = threading.local()

    @property
    def name(self) -> str:
        return "openai-compatible"

    def _client_or_raise(self) -> OpenAICompatibleClient:
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = OpenAICompatibleClient(
                api_base_url=self.config.api_base_url,
                api_key=self.api_key,
                timeout_seconds=self.config.api_timeout_seconds,
            )
            self._thread_local.client = client
        return client

    def execute(self, request: ExecutionRequest, attempt_number: int) -> ExecutionResult:
        try:
            result = self._client_or_raise().chat_completion(
                model=self.config.model,
                user_prompt=request.job.prompt,
                max_tokens=self.config.api_max_tokens,
                temperature=self.config.api_temperature,
                seed=request.job.generator_seed,
            )
        except OpenAICompatibleTransientError as exc:
            raise TransientExecutionError(str(exc)) from exc
        except OpenAICompatiblePermanentError as exc:
            raise PermanentExecutionError(str(exc)) from exc

        rendered_text = result.text.strip()
        if not rendered_text:
            raise TransientExecutionError("provider returned empty rendered_text")
        rendered_text, repaired_evidence_id_count = normalize_evidence_ids_to_source_ids(
            request.job,
            rendered_text,
        )
        validate_rendered_representation(request.job, rendered_text)

        metadata = {
            "api_response_id": result.response_id,
            "api_response_model": result.response_model,
            "api_usage": result.usage,
            "api_attempt_number": attempt_number,
            "repaired_evidence_id_count": repaired_evidence_id_count,
        }
        return ExecutionResult(rendered_text=rendered_text, metadata=metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Prompt-job JSONL file(s) or directories produced by render_treatments.py",
    )
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--status-log-path",
        default="",
        help="JSONL status log path. Defaults next to the output path.",
    )
    parser.add_argument(
        "--mode",
        choices=("dry-run", "mock", "openai-compatible"),
        default="mock",
        help="Execution mode. `dry-run` validates and logs; `mock` renders deterministic summaries; `openai-compatible` calls a chat-completions API.",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_MOCK_PROVIDER,
        help="Provider label recorded in metadata. The current implementation uses the mock executor.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MOCK_MODEL,
        help="Model label recorded in metadata.",
    )
    parser.add_argument(
        "--upstream-model-family",
        default="",
        help=(
            "Canonical upstream representation model family recorded in outputs. "
            "Prefer setting this explicitly; when omitted only known safe model aliases are derived."
        ),
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
        default=900,
        help="Suggested max_tokens for openai-compatible representation generation.",
    )
    parser.add_argument(
        "--api-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for openai-compatible representation generation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum retries after the first failed attempt.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=0.0,
        help="Optional sleep interval between retry attempts.",
    )
    parser.add_argument(
        "--mock-retry-mod",
        type=int,
        default=0,
        help="If > 0, deterministically force a transient failure on first attempt for matching jobs.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=0,
        help="Optional cap on the number of parsed input jobs.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing completed rows in the output JSONL.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent representation jobs to execute.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Deterministically partition jobs into this many shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to execute when --shard-count > 1.",
    )
    return parser.parse_args()


def default_status_log_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.status.jsonl")
    return output_path.with_name(f"{output_path.name}.status.jsonl")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shard_bucket(key: str, shard_count: int) -> int:
    return int(stable_digest(key)[:16], 16) % shard_count


def job_matches_shard(job_key: str, shard_count: int, shard_index: int) -> bool:
    if shard_count <= 1:
        return True
    return shard_bucket(job_key, shard_count) == shard_index


def stable_int_seed(*parts: Any) -> int:
    material = "||".join(str(part) for part in parts)
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def derive_upstream_model_family_from_model(model: str) -> str | None:
    normalized = model.strip()
    lowered = normalized.lower()
    if not normalized:
        return None
    if lowered == DEFAULT_MOCK_MODEL or lowered.startswith("mock-"):
        return "mock"
    if lowered in {"gpt-5.2", "claude-sonnet-4.5", "qwen3-235b-a22b", "deepseek-v3.1"}:
        return lowered
    if lowered.startswith("gpt-5.2"):
        return "gpt-5.2"
    if lowered.startswith("claude-sonnet-4-5") or lowered.startswith("claude-sonnet-4.5"):
        return "claude-sonnet-4.5"
    if lowered.startswith("qwen/qwen3-235b-a22b") or lowered.startswith("qwen3-235b-a22b"):
        return "qwen3-235b-a22b"
    if lowered.startswith("deepseek/deepseek-chat-v3.1") or lowered.startswith("deepseek-v3.1"):
        return "deepseek-v3.1"
    return None


def resolve_upstream_model_family(value: str, model: str) -> str:
    explicit = value.strip()
    if explicit:
        return explicit
    derived = derive_upstream_model_family_from_model(model)
    if derived is not None:
        return derived
    raise ValueError(
        "--upstream-model-family is required for this model; only known safe model aliases "
        "can be derived automatically"
    )


def completed_key(job_key: str, upstream_model_family: str) -> str:
    return json.dumps(
        {
            "job_key": job_key.strip(),
            "upstream_model_family": upstream_model_family.strip(),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def upstream_model_family_from_output_record(record: dict[str, Any]) -> str:
    value = record.get("upstream_model_family")
    if is_nonempty_string(value):
        return str(value).strip()
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        metadata_value = metadata.get("upstream_model_family")
        if is_nonempty_string(metadata_value):
            return str(metadata_value).strip()
        model = metadata.get("model")
        if is_nonempty_string(model):
            derived = derive_upstream_model_family_from_model(str(model))
            if derived is not None:
                return derived
    model = record.get("model")
    if is_nonempty_string(model):
        derived = derive_upstream_model_family_from_model(str(model))
        if derived is not None:
            return derived
    return LEGACY_UNKNOWN_UPSTREAM_MODEL_FAMILY


def json_dump_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def append_jsonl_line(handle: TextIO, payload: dict[str, Any]) -> None:
    handle.write(json_dump_line(payload))
    handle.write("\n")
    handle.flush()


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_int_like(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def normalize_string(value: str) -> str:
    return " ".join(value.strip().split())


def truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def sanitize_claim_text(value: str, allow_identity: bool = False) -> str:
    cleaned = normalize_string(value)
    if not allow_identity:
        cleaned = EXACT_DATE_PATTERN.sub("masked_date", cleaned)
        cleaned = TICKER_PATTERN.sub("masked_ticker", cleaned)
        cleaned = MASKED_COMPANY_PATTERN.sub("the company", cleaned)
    return truncate_text(cleaned, 220)


def format_source_ids(source_ids: Any) -> str:
    if isinstance(source_ids, list):
        parts = [str(item).strip() for item in source_ids if str(item).strip()]
        if parts:
            return ", ".join(parts)
    return "source_id_missing"


def derive_job_identity(record: dict[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "event_id": record.get("event_id"),
        "treatment": record.get("treatment"),
        "representation_seed": record.get("representation_seed"),
        "generator_seed": record.get("generator_seed"),
    }
    profile_id = record.get("profile_id")
    if profile_id is not None:
        identity["profile_id"] = profile_id
    return identity


def derive_job_key(record: dict[str, Any]) -> str:
    identity = derive_job_identity(record)
    packed = json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return stable_digest(packed)


def parse_prompt_job(record: Any, source_path: Path, line_number: int) -> PromptJob:
    if not isinstance(record, dict):
        raise PromptJobError("row must be a JSON object")
    event_id = record.get("event_id")
    treatment = record.get("treatment")
    representation_seed = record.get("representation_seed")
    generator_seed = record.get("generator_seed")
    prompt = record.get("prompt")
    profile_id = record.get("profile_id")
    evidence_bank_path = record.get("evidence_bank_path")

    if not is_nonempty_string(event_id):
        raise PromptJobError("missing non-empty event_id")
    if not is_nonempty_string(treatment):
        raise PromptJobError("missing non-empty treatment")
    if not is_int_like(representation_seed):
        raise PromptJobError("representation_seed must be an integer")
    if not is_int_like(generator_seed):
        raise PromptJobError("generator_seed must be an integer")
    if not is_nonempty_string(prompt):
        raise PromptJobError("missing non-empty prompt")
    if profile_id is not None and not is_nonempty_string(profile_id):
        raise PromptJobError("profile_id must be a non-empty string when present")
    if treatment == "T3_independent_summary" and not is_nonempty_string(profile_id):
        raise PromptJobError("T3_independent_summary rows require profile_id")

    clean_record = dict(record)
    clean_record["event_id"] = event_id.strip()
    clean_record["treatment"] = treatment.strip()
    clean_record["representation_seed"] = int(representation_seed)
    clean_record["generator_seed"] = int(generator_seed)
    clean_record["prompt"] = str(prompt)
    if profile_id is not None:
        clean_record["profile_id"] = profile_id.strip()

    return PromptJob(
        source_path=source_path,
        line_number=line_number,
        event_id=clean_record["event_id"],
        treatment=clean_record["treatment"],
        representation_seed=clean_record["representation_seed"],
        generator_seed=clean_record["generator_seed"],
        prompt=clean_record["prompt"],
        profile_id=clean_record.get("profile_id"),
        evidence_bank_path=str(evidence_bank_path) if evidence_bank_path is not None else None,
        raw_record=clean_record,
        job_key=derive_job_key(clean_record),
        prompt_sha256=stable_digest(clean_record["prompt"]),
    )


def iter_prompt_input_paths(items: Iterable[str]) -> list[Path]:
    resolved: list[Path] = []
    for item in items:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            resolved.append(path)
            continue

        prompt_job_files = sorted(path.rglob("prompt_jobs*.jsonl"))
        if prompt_job_files:
            resolved.extend(prompt_job_files)
            continue

        generic_jsonl = sorted(path.rglob("*.jsonl"))
        if generic_jsonl:
            resolved.extend(generic_jsonl)
            continue

        raise FileNotFoundError(f"no JSONL files found under {path}")

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        canonical = path.resolve()
        if canonical in seen:
            continue
        seen.add(canonical)
        unique_paths.append(canonical)
    return unique_paths


def load_completed_job_keys(output_path: Path) -> tuple[set[str], int, int, int]:
    if not output_path.exists():
        return set(), 0, 0, 0

    completed: set[str] = set()
    parsed_rows = 0
    invalid_rows = 0
    duplicate_rows = 0
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue
            if not isinstance(record, dict):
                invalid_rows += 1
                continue
            parsed_rows += 1
            job_key = record.get("job_key")
            if is_nonempty_string(job_key):
                normalized_key = completed_key(
                    str(job_key),
                    upstream_model_family_from_output_record(record),
                )
                if normalized_key in completed:
                    duplicate_rows += 1
                completed.add(normalized_key)
                continue
            try:
                normalized_key = completed_key(
                    derive_job_key(record),
                    upstream_model_family_from_output_record(record),
                )
                if normalized_key in completed:
                    duplicate_rows += 1
                completed.add(normalized_key)
            except Exception:
                invalid_rows += 1
    return completed, parsed_rows, invalid_rows, duplicate_rows


def extract_evidence_units(prompt: str) -> list[dict[str, Any]]:
    if PROMPT_BANK_MARKER not in prompt:
        return []
    payload = prompt.split(PROMPT_BANK_MARKER, 1)[1]
    units: list[dict[str, Any]] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            units.append(record)
    return units


def build_evidence_id_to_source_ids(prompt: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for unit in extract_evidence_units(prompt):
        evidence_id = str(unit.get("evidence_id", "")).strip()
        raw_source_ids = unit.get("source_ids", [])
        if not isinstance(raw_source_ids, list):
            continue
        source_ids = [
            source_id.strip()
            for source_id in raw_source_ids
            if is_nonempty_string(source_id)
        ]
        if evidence_id and source_ids:
            mapping[evidence_id] = source_ids
    return mapping


def normalize_evidence_ids_to_source_ids(job: PromptJob, rendered_text: str) -> tuple[str, int]:
    mapping = build_evidence_id_to_source_ids(job.prompt)
    if not mapping:
        return rendered_text, 0

    replacements = 0

    def replacer(match: re.Match[str]) -> str:
        nonlocal replacements
        evidence_id = match.group(0)
        source_ids = mapping.get(evidence_id)
        if not source_ids:
            return evidence_id
        replacements += 1
        return ", ".join(source_ids)

    normalized = EVIDENCE_ID_PATTERN.sub(replacer, rendered_text)
    return normalized, replacements


def render_mock_representation(job: PromptJob) -> tuple[str, int, int, int]:
    units = extract_evidence_units(job.prompt)
    allow_identity = "visibility_policy: entity_visible_outcome_blind" in job.prompt
    derived_seed = stable_int_seed(job.generator_seed, job.prompt_sha256, job.job_key)
    rng = random.Random(derived_seed)

    if not units:
        fallback = (
            "Mock summary placeholder. The harness validated the prompt shell, "
            "but it could not parse canonical evidence-unit rows from the prompt body."
        )
        return fallback, derived_seed, 0, 0

    grouped: dict[str, list[dict[str, Any]]] = {}
    for unit in units:
        category = str(unit.get("category", "other")).strip() or "other"
        grouped.setdefault(category, []).append(unit)

    categories = sorted(grouped)
    rng.shuffle(categories)

    sections = [rng.choice(INTRO_TEMPLATES)]
    for category in categories:
        category_units = list(grouped[category])
        rng.shuffle(category_units)
        selected_units = category_units[:3]
        claims = "; ".join(
            format_unit_claim(unit, allow_identity=allow_identity)
            for unit in selected_units
        )
        label = CATEGORY_LABELS.get(category, category.replace("_", " ").title())
        template = rng.choice(CATEGORY_TEMPLATES)
        sections.append(
            template.format(
                label=label,
                label_lower=label.lower(),
                claims=claims,
            )
        )
        omitted_count = max(0, len(category_units) - len(selected_units))
        if omitted_count:
            sections.append(
                f"Mock mode omits {omitted_count} additional {label.lower()} items for brevity."
            )
    sections.append(rng.choice(CLOSING_TEMPLATES))
    return " ".join(section for section in sections if section), derived_seed, len(units), len(categories)


def format_unit_claim(unit: dict[str, Any], allow_identity: bool = False) -> str:
    claim = sanitize_claim_text(
        str(unit.get("claim", "undisclosed claim")),
        allow_identity=allow_identity,
    )
    value = unit.get("value")
    if value not in ("", None):
        claim = f"{claim} (value={value})"
    citations = format_source_ids(unit.get("source_ids"))
    return f"{claim} [{citations}]"


def validate_rendered_representation(job: PromptJob, rendered_text: str) -> None:
    if EVIDENCE_ID_PATTERN.search(rendered_text):
        raise TransientExecutionError(
            "provider cited evidence IDs (E_SRC/E_XBRL/E_CTX) instead of source IDs"
        )
    if not SOURCE_ID_PATTERN.search(rendered_text):
        raise TransientExecutionError("provider response did not include any source IDs")


def build_executor(config: ExecutionConfig) -> RepresentationExecutor:
    if config.mode == "mock":
        return MockRepresentationExecutor(config)
    if config.mode == "openai-compatible":
        return OpenAICompatibleRepresentationExecutor(config)
    raise PermanentExecutionError(f"unsupported execution mode: {config.mode}")


def make_status_payload(
    *,
    status: str,
    message: str,
    source_path: Path,
    line_number: int,
    job: PromptJob | None = None,
    attempt_number: int | None = None,
    retry_count: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "status": status,
        "message": message,
        "source_path": str(source_path),
        "line_number": line_number,
    }
    if job is not None:
        payload.update(
            {
                "job_key": job.job_key,
                "event_id": job.event_id,
                "treatment": job.treatment,
                "representation_seed": job.representation_seed,
                "generator_seed": job.generator_seed,
                "upstream_model_family": extra["upstream_model_family"]
                if extra and "upstream_model_family" in extra
                else None,
            }
        )
        if payload["upstream_model_family"] is None:
            del payload["upstream_model_family"]
        if job.profile_id is not None:
            payload["profile_id"] = job.profile_id
    if attempt_number is not None:
        payload["attempt_number"] = attempt_number
    if retry_count is not None:
        payload["retry_count"] = retry_count
    if extra:
        payload.update(extra)
    return payload


def build_output_row(
    job: PromptJob,
    result: ExecutionResult,
    config: ExecutionConfig,
    executor: RepresentationExecutor,
    attempt_count: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "job_key": job.job_key,
        "event_id": job.event_id,
        "treatment": job.treatment,
        "representation_seed": job.representation_seed,
        "generator_seed": job.generator_seed,
        "upstream_model_family": config.upstream_model_family,
        "rendered_text": result.rendered_text,
        "metadata": {
            "execution_mode": config.mode,
            "provider": config.provider,
            "model": config.model,
            "upstream_model_family": config.upstream_model_family,
            "executor": executor.name,
            "attempt_count": attempt_count,
            "retry_count": max(0, attempt_count - 1),
            "prompt_sha256": job.prompt_sha256,
            "rendered_text_sha256": stable_digest(result.rendered_text),
            "prompt_job_path": str(job.source_path),
            "prompt_job_line_number": job.line_number,
            **result.metadata,
        },
    }
    if job.profile_id is not None:
        row["profile_id"] = job.profile_id
    if job.evidence_bank_path is not None:
        row["evidence_bank_path"] = job.evidence_bank_path
    return row


def iter_input_lines(path: Path) -> Iterable[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            yield line_number, stripped


def maybe_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def execute_job(
    job: PromptJob,
    executor: RepresentationExecutor,
    config: ExecutionConfig,
) -> JobExecutionOutcome:
    max_attempts = 1 + max(0, config.max_retries)
    attempt_number = 0
    retry_notices: list[RetryNotice] = []
    while attempt_number < max_attempts:
        attempt_number += 1
        try:
            result = executor.execute(ExecutionRequest(job=job, config=config), attempt_number)
            return JobExecutionOutcome(
                job=job,
                result=result,
                attempts=attempt_number,
                retry_notices=tuple(retry_notices),
                error_message=None,
            )
        except TransientExecutionError as exc:
            if attempt_number >= max_attempts:
                return JobExecutionOutcome(
                    job=job,
                    result=None,
                    attempts=attempt_number,
                    retry_notices=tuple(retry_notices),
                    error_message=str(exc),
                )
            retry_notices.append(
                RetryNotice(
                    attempt_number=attempt_number,
                    retry_count=attempt_number,
                    message=str(exc),
                )
            )
            maybe_sleep(config.retry_backoff_seconds)
        except PermanentExecutionError as exc:
            return JobExecutionOutcome(
                job=job,
                result=None,
                attempts=attempt_number,
                retry_notices=tuple(retry_notices),
                error_message=str(exc),
            )
    return JobExecutionOutcome(
        job=job,
        result=None,
        attempts=max_attempts,
        retry_notices=tuple(retry_notices),
        error_message="execution loop exited unexpectedly",
    )


def record_job_outcome(
    *,
    outcome: JobExecutionOutcome,
    summary: RunSummary,
    status_handle: TextIO,
    output_handle: TextIO | None,
    config: ExecutionConfig,
    executor: RepresentationExecutor,
    completed_keys: set[str],
) -> None:
    job = outcome.job
    for retry_notice in outcome.retry_notices:
        append_jsonl_line(
            status_handle,
            make_status_payload(
                status="retry_scheduled",
                message=retry_notice.message,
                source_path=job.source_path,
                line_number=job.line_number,
                job=job,
                attempt_number=retry_notice.attempt_number,
                retry_count=retry_notice.retry_count,
                extra={"upstream_model_family": config.upstream_model_family},
            ),
        )

    summary.total_attempts += outcome.attempts
    summary.total_retries += max(0, outcome.attempts - 1)
    if outcome.attempts > 1:
        summary.retried_jobs += 1

    if not outcome.success:
        summary.failed_rows += 1
        append_jsonl_line(
            status_handle,
            make_status_payload(
                status="failed",
                message=outcome.error_message or "unknown execution failure",
                source_path=job.source_path,
                line_number=job.line_number,
                job=job,
                attempt_number=outcome.attempts,
                retry_count=max(0, outcome.attempts - 1),
                extra={"upstream_model_family": config.upstream_model_family},
            ),
        )
        return

    summary.completed_rows += 1
    if outcome.attempts > 1:
        append_jsonl_line(
            status_handle,
            make_status_payload(
                status="retried_success",
                message="row completed after retry",
                source_path=job.source_path,
                line_number=job.line_number,
                job=job,
                attempt_number=outcome.attempts,
                retry_count=outcome.attempts - 1,
                extra={"upstream_model_family": config.upstream_model_family},
            ),
        )
    else:
        append_jsonl_line(
            status_handle,
            make_status_payload(
                status="completed",
                message="row completed",
                source_path=job.source_path,
                line_number=job.line_number,
                job=job,
                attempt_number=outcome.attempts,
                retry_count=0,
                extra={"upstream_model_family": config.upstream_model_family},
            ),
        )
    assert output_handle is not None
    assert outcome.result is not None
    append_jsonl_line(
        output_handle,
        build_output_row(job, outcome.result, config, executor, outcome.attempts),
    )
    completed_keys.add(completed_key(job.job_key, config.upstream_model_family))


def process_jobs(
    *,
    input_paths: list[Path],
    output_path: Path,
    status_log_path: Path,
    config: ExecutionConfig,
    resume: bool,
    max_jobs: int,
    shard_count: int,
    shard_index: int,
) -> RunSummary:
    summary = RunSummary(input_files=len(input_paths))
    completed_keys: set[str] = set()
    if resume:
        completed_keys, parsed_rows, invalid_rows, duplicate_rows = load_completed_job_keys(
            output_path
        )
        summary.existing_output_rows = parsed_rows
        summary.existing_output_invalid = invalid_rows
        summary.existing_output_duplicates = duplicate_rows
        if invalid_rows:
            raise ValueError(
                f"existing output contains {invalid_rows} invalid row(s): {output_path}. "
                "Use a fresh output path for a new run."
            )
        if duplicate_rows:
            raise ValueError(
                f"existing output contains {duplicate_rows} duplicate job_key/upstream_model_family row(s): "
                f"{output_path}. Deduplicate it or use a fresh output path before resuming."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_log_path.parent.mkdir(parents=True, exist_ok=True)
    executor = build_executor(config) if config.mode != "dry-run" else None
    queued_jobs: list[PromptJob] = []
    reserved_job_keys = set(completed_keys) if resume else set()
    stop_parsing = False

    output_handle: TextIO | None = None
    if config.mode != "dry-run":
        output_handle = output_path.open("a", encoding="utf-8")

    with status_log_path.open("a", encoding="utf-8") as status_handle:
        try:
            for input_path in input_paths:
                for line_number, raw_line in iter_input_lines(input_path):
                    summary.seen_rows += 1
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        summary.invalid_rows += 1
                        append_jsonl_line(
                            status_handle,
                            make_status_payload(
                                status="invalid_input",
                                message=f"invalid JSON: {exc.msg} (column {exc.colno})",
                                source_path=input_path,
                                line_number=line_number,
                                extra={"raw_preview": raw_line[:200]},
                            ),
                        )
                        continue
                    try:
                        job = parse_prompt_job(record, input_path, line_number)
                    except PromptJobError as exc:
                        summary.invalid_rows += 1
                        append_jsonl_line(
                            status_handle,
                            make_status_payload(
                                status="invalid_input",
                                message=str(exc),
                                source_path=input_path,
                                line_number=line_number,
                                extra={"raw_preview": str(record)[:200]},
                            ),
                        )
                        continue

                    if not job_matches_shard(job.job_key, shard_count, shard_index):
                        continue

                    if max_jobs > 0 and summary.selected_rows >= max_jobs:
                        stop_parsing = True
                        break

                    summary.selected_rows += 1

                    job_completed_key = completed_key(job.job_key, config.upstream_model_family)

                    if job_completed_key in reserved_job_keys:
                        summary.skipped_existing += 1
                        append_jsonl_line(
                            status_handle,
                            make_status_payload(
                                status="skipped_existing",
                                message="row already present in output or queued earlier in this run",
                                source_path=input_path,
                                line_number=line_number,
                                job=job,
                                extra={"upstream_model_family": config.upstream_model_family},
                            ),
                        )
                        continue

                    if config.mode == "dry-run":
                        summary.dry_run_rows += 1
                        append_jsonl_line(
                            status_handle,
                            make_status_payload(
                                status="dry_run_planned",
                                message="row validated and queued for future execution",
                                source_path=input_path,
                                line_number=line_number,
                                job=job,
                                extra={"upstream_model_family": config.upstream_model_family},
                            ),
                        )
                        continue

                    reserved_job_keys.add(job_completed_key)
                    queued_jobs.append(job)

                if stop_parsing:
                    break

            if config.mode != "dry-run":
                assert executor is not None
                if config.workers <= 1:
                    for job in queued_jobs:
                        record_job_outcome(
                            outcome=execute_job(job, executor, config),
                            summary=summary,
                            status_handle=status_handle,
                            output_handle=output_handle,
                            config=config,
                            executor=executor,
                            completed_keys=completed_keys,
                        )
                else:
                    with ThreadPoolExecutor(max_workers=config.workers) as pool:
                        future_to_job = {
                            pool.submit(execute_job, job, executor, config): job
                            for job in queued_jobs
                        }
                        for future in as_completed(future_to_job):
                            job = future_to_job[future]
                            try:
                                outcome = future.result()
                            except Exception as exc:
                                outcome = JobExecutionOutcome(
                                    job=job,
                                    result=None,
                                    attempts=1,
                                    retry_notices=(),
                                    error_message=f"unexpected worker error: {exc}",
                                )
                            record_job_outcome(
                                outcome=outcome,
                                summary=summary,
                                status_handle=status_handle,
                                output_handle=output_handle,
                                config=config,
                                executor=executor,
                                completed_keys=completed_keys,
                            )
        finally:
            if output_handle is not None:
                output_handle.close()

    return summary


def print_summary(
    summary: RunSummary,
    *,
    mode: str,
    output_path: Path,
    status_log_path: Path,
    resume: bool,
    shard_count: int,
    shard_index: int,
) -> None:
    print("representation harness summary:")
    print(f"  mode: {mode}")
    print(f"  input_files: {summary.input_files}")
    print(f"  seen_rows: {summary.seen_rows}")
    print(f"  selected_rows: {summary.selected_rows}")
    if shard_count > 1:
        print(f"  shard: {shard_index}/{shard_count}")
    if resume:
        print(f"  existing_output_rows: {summary.existing_output_rows}")
        if summary.existing_output_invalid:
            print(f"  existing_output_invalid: {summary.existing_output_invalid}")
        if summary.existing_output_duplicates:
            print(f"  existing_output_duplicates: {summary.existing_output_duplicates}")
    print(f"  invalid_rows: {summary.invalid_rows}")
    print(f"  skipped_existing: {summary.skipped_existing}")
    if mode == "dry-run":
        print(f"  planned_rows: {summary.dry_run_rows}")
    else:
        print(f"  completed_rows: {summary.completed_rows}")
        print(f"  failed_rows: {summary.failed_rows}")
        print(f"  total_attempts: {summary.total_attempts}")
        print(f"  total_retries: {summary.total_retries}")
        print(f"  retried_jobs: {summary.retried_jobs}")
    print(f"  output_path: {output_path}")
    print(f"  status_log_path: {status_log_path}")


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_path).resolve()
    status_log_path = (
        Path(args.status_log_path).resolve()
        if args.status_log_path
        else default_status_log_path(output_path).resolve()
    )
    if output_path == status_log_path:
        raise ValueError("output_path and status_log_path must differ")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")
    if args.retry_backoff_seconds < 0:
        raise ValueError("--retry-backoff-seconds must be >= 0")
    if args.mock_retry_mod < 0:
        raise ValueError("--mock-retry-mod must be >= 0")
    if args.max_jobs < 0:
        raise ValueError("--max-jobs must be >= 0")
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

    config = ExecutionConfig(
        mode=args.mode,
        provider=args.provider,
        model=args.model,
        upstream_model_family=resolve_upstream_model_family(
            args.upstream_model_family,
            args.model,
        ),
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        mock_retry_mod=args.mock_retry_mod,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        api_key_env_var=args.api_key_env_var,
        api_timeout_seconds=args.api_timeout_seconds,
        api_max_tokens=args.api_max_tokens,
        api_temperature=args.api_temperature,
        workers=args.workers,
    )
    input_paths = iter_prompt_input_paths(args.inputs)
    summary = process_jobs(
        input_paths=input_paths,
        output_path=output_path,
        status_log_path=status_log_path,
        config=config,
        resume=not args.no_resume,
        max_jobs=args.max_jobs,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    print_summary(
        summary,
        mode=config.mode,
        output_path=output_path,
        status_log_path=status_log_path,
        resume=not args.no_resume,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    if summary.invalid_rows or summary.failed_rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
