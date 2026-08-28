#!/usr/bin/env python3
"""Compute event-level diversity metrics from validated decision outputs.

The script reads validated downstream decision outputs, optionally joins hidden
post-event outcomes, and emits event-level cell metrics plus paired T2-vs-T3
contrast rows for later statistical analysis.

Outputs:
- decision_rows.csv
- cell_metrics.csv
- t2_t3_contrasts.csv
- treatment_contrasts.csv
- run_summary.json

The rationale-distance metric uses a local sentence-transformer model only when
the caller supplies one. Otherwise it falls back to a deterministic lexical
cosine-distance proxy over token and character n-gram counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "decisions.jsonl"
JSON_SUFFIXES = {".json", ".jsonl"}

TREATMENT_FAMILIES = {
    "T1_full": "T1",
    "T1_length_matched": "T1",
    "T1_raw_public_information": "T1",
    "T2_shared_summary": "T2",
    "T2_shared_llm_summary": "T2",
    "T3_independent": "T3",
    "T3_independent_summary": "T3",
    "T3_independent_summaries": "T3",
    "T3_independent_llm_summary": "T3",
    "T4_structured": "T4",
    "T4_structured_evidence_card": "T4",
    "T4_shared_atomic_evidence_view_control": "T4",
    "T4_full_structured_evidence_ledger": "T4",
    "T4_SAEV_deterministic": "T4",
    "B0_canonical_evidence_only": "B0",
    "T5_linguistic_deframing": "T5",
    "T6_canonical_evidence_order_randomized": "T6",
}

PROFILE_GROUP_BY_PROFILE = {
    "day_trader": "retail",
    "retail_day_trader": "retail",
    "swing_trader": "retail",
    "retail_swing_trader": "retail",
    "long_term_retail": "retail",
    "retail_long_term_fundamental": "retail",
    "event_driven_hedge_fund": "institutional",
    "institutional_event_driven_hedge_fund": "institutional",
    "prop_trading": "institutional",
    "institutional_prop_trader": "institutional",
    "institutional_prop_trading": "institutional",
    "investment_advisor": "institutional",
    "institutional_investment_advisor": "institutional",
    "unprofiled_baseline": "unprofiled",
}

CANONICAL_PROFILE_BY_PROFILE = {
    "day_trader": "day_trader",
    "retail_day_trader": "day_trader",
    "swing_trader": "swing_trader",
    "retail_swing_trader": "swing_trader",
    "long_term_retail": "long_term_retail",
    "retail_long_term_fundamental": "long_term_retail",
    "event_driven_hedge_fund": "event_driven_hedge_fund",
    "institutional_event_driven_hedge_fund": "event_driven_hedge_fund",
    "prop_trading": "prop_trading",
    "institutional_prop_trader": "prop_trading",
    "institutional_prop_trading": "prop_trading",
    "investment_advisor": "investment_advisor",
    "institutional_investment_advisor": "investment_advisor",
    "unprofiled_baseline": "unprofiled_baseline",
}

EXPECTED_PROFILES_BY_INVESTOR_GROUP = {
    "all_profiles": (
        "day_trader",
        "swing_trader",
        "long_term_retail",
        "event_driven_hedge_fund",
        "prop_trading",
        "investment_advisor",
    ),
    "retail": (
        "day_trader",
        "swing_trader",
        "long_term_retail",
    ),
    "institutional": (
        "event_driven_hedge_fund",
        "prop_trading",
        "investment_advisor",
    ),
    "unprofiled": ("unprofiled_baseline",),
}

INVESTOR_GROUPS = ("all_profiles", "retail", "institutional", "unprofiled")
ACTION_CODES = {"buy": 1, "hold": 0, "sell": -1}
SIGN_EPSILON = 1e-12
LEGACY_UPSTREAM_MODEL_FAMILY = "legacy_unspecified"

CELL_METRIC_NAMES = (
    "std_expected_return_5d",
    "mean_pairwise_abs_diff_expected_return_5d",
    "iqr_expected_return_5d",
    "confidence_dispersion",
    "entropy_buy_hold_sell",
    "hhi_buy_hold_sell",
    "action_diversity_one_minus_hhi",
    "absolute_net_direction",
    "action_direction_diversity_one_minus_abs_net_direction",
    "mean_pairwise_sentence_embedding_distance",
    "source_id_overlap",
    "source_id_diversity_one_minus_overlap",
    "evidence_category_overlap",
    "evidence_category_diversity_one_minus_overlap",
)

CONTRAST_METRIC_NAMES = (
    "std_expected_return_5d",
    "mean_pairwise_abs_diff_expected_return_5d",
    "iqr_expected_return_5d",
    "confidence_dispersion",
    "entropy_buy_hold_sell",
    "action_diversity_one_minus_hhi",
    "action_direction_diversity_one_minus_abs_net_direction",
    "mean_pairwise_sentence_embedding_distance",
    "source_id_diversity_one_minus_overlap",
    "evidence_category_diversity_one_minus_overlap",
)

EVENT_METADATA_FIELDS = (
    "ticker",
    "company",
    "event_date",
    "market_cap_group",
    "sector",
)

METRIC_UNITS = {
    "std_expected_return_5d": "decimal_return",
    "mean_pairwise_abs_diff_expected_return_5d": "decimal_return",
    "iqr_expected_return_5d": "decimal_return",
    "confidence_dispersion": "unitless",
    "entropy_buy_hold_sell": "bits",
    "hhi_buy_hold_sell": "unitless",
    "action_diversity_one_minus_hhi": "unitless",
    "absolute_net_direction": "unitless",
    "action_direction_diversity_one_minus_abs_net_direction": "unitless",
    "mean_pairwise_sentence_embedding_distance": "unitless",
    "source_id_overlap": "unitless",
    "source_id_diversity_one_minus_overlap": "unitless",
    "evidence_category_overlap": "unitless",
    "evidence_category_diversity_one_minus_overlap": "unitless",
}

METRIC_GROUPS = {
    "std_expected_return_5d": "belief",
    "mean_pairwise_abs_diff_expected_return_5d": "belief",
    "iqr_expected_return_5d": "belief",
    "confidence_dispersion": "belief",
    "entropy_buy_hold_sell": "action",
    "action_diversity_one_minus_hhi": "action",
    "action_direction_diversity_one_minus_abs_net_direction": "action",
    "mean_pairwise_sentence_embedding_distance": "rationale",
    "source_id_diversity_one_minus_overlap": "rationale",
    "evidence_category_diversity_one_minus_overlap": "rationale",
}

ASSUMPTIONS = {
    "std_definition": "population standard deviation within each event-level cell",
    "iqr_definition": "p75 - p25 using linear interpolation on sorted values",
    "confidence_dispersion_definition": "population standard deviation of confidence",
    "rationale_text_definition": "all key_reasons joined into one rationale string per decision row",
    "pairwise_metrics_rule": "pairwise metrics are null when a cell has fewer than 2 profiles",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute event-level diversity metrics and paired T2-vs-T3 contrasts "
            "from validated decision outputs."
        )
    )
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output artifacts. Defaults next to the input file.",
    )
    parser.add_argument(
        "--hidden-outcomes-csv",
        default=None,
        help="Optional hidden outcomes CSV with at least event_id and CAR_1_5.",
    )
    parser.add_argument(
        "--event-metadata-csv",
        default=None,
        help=(
            "Optional event metadata CSV with event_id plus columns such as "
            "ticker, event_date, and market_cap_group."
        ),
    )
    parser.add_argument(
        "--evidence-banks",
        default=None,
        help=(
            "Optional evidence-unit-bank JSON/JSONL file or directory for "
            "evidence category mapping."
        ),
    )
    parser.add_argument(
        "--source-packets",
        default=None,
        help=(
            "Optional source-packet JSON/JSONL file or directory. Used only as a "
            "fallback evidence-category mapping source when evidence banks are absent."
        ),
    )
    parser.add_argument(
        "--text-distance-method",
        choices=("auto", "token_cosine_proxy", "sentence_transformer"),
        default="auto",
        help=(
            "Rationale-distance backend. 'auto' uses a local sentence-transformer "
            "only when --embedding-model is supplied; otherwise it uses the "
            "deterministic proxy."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Optional local sentence-transformer model path or name. The script "
            "does not download models."
        ),
    )
    parser.add_argument(
        "--sci-near-zero-threshold",
        type=float,
        default=1e-6,
        help=(
            "Denominator threshold below which SCI is marked unstable and left null."
        ),
    )
    return parser.parse_args()


def default_output_dir(input_path: Path) -> Path:
    stem = input_path.stem if input_path.suffix else input_path.name
    return input_path.with_name(f"{stem}.diversity_metrics")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_list_of_strings(value: Any) -> bool:
    return isinstance(value, list) and all(is_nonempty_string(item) for item in value)


def normalize_string(value: Any) -> str | None:
    if not is_nonempty_string(value):
        return None
    return str(value).strip()


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif is_nonempty_string(value):
        try:
            numeric = float(str(value).strip())
        except ValueError:
            return None
    else:
        return None
    return numeric if math.isfinite(numeric) else None


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if is_nonempty_string(value):
        text = str(value).strip()
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
    return None


def normalize_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    normalized: list[str] = []
    for item in value:
        item_text = normalize_string(item)
        if item_text is None:
            continue
        normalized.append(item_text)
    return normalized


def upstream_model_family_from_record(
    record: dict[str, Any],
    treatment_family: str | None,
) -> str:
    upstream_family = normalize_string(record.get("upstream_model_family"))
    if upstream_family is not None:
        return upstream_family

    request_metadata = record.get("request_metadata")
    if isinstance(request_metadata, dict):
        passthrough_fields = request_metadata.get("passthrough_fields")
        if isinstance(passthrough_fields, dict):
            upstream_family = normalize_string(
                passthrough_fields.get("upstream_model_family")
            )
            if upstream_family is not None:
                return upstream_family

    if treatment_family == "T1":
        return "none"
    if treatment_family == "B0":
        return "deterministic_canonical_evidence"
    if treatment_family == "T5":
        return "deterministic_linguistic_deframing"
    if treatment_family == "T6":
        return "deterministic_evidence_order_randomization"
    return LEGACY_UPSTREAM_MODEL_FAMILY


def infer_treatment_family(treatment: str | None) -> str | None:
    if treatment is None:
        return None
    if treatment in TREATMENT_FAMILIES:
        return TREATMENT_FAMILIES[treatment]
    prefix = treatment.split("_", 1)[0]
    if prefix in {"T1", "T2", "T3", "T4", "T5", "T6", "B0"}:
        return prefix
    return None


def summarize_value(value: Any, max_length: int = 160) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_length else value[: max_length - 3] + "..."
    if isinstance(value, list):
        preview = value[:5]
        if len(value) > 5:
            preview.append(f"... ({len(value)} items total)")
        return preview
    if isinstance(value, dict):
        preview_items = list(value.items())[:5]
        preview = {key: val for key, val in preview_items}
        if len(value) > 5:
            preview["..."] = f"{len(value)} keys total"
        return preview
    return repr(value)


def load_jsonl_records(text: str) -> tuple[list[tuple[int, Any]], list[dict[str, Any]]]:
    rows: list[tuple[int, Any]] = []
    errors: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            rows.append((line_number, json.loads(stripped)))
        except json.JSONDecodeError as exc:
            errors.append(
                {
                    "line_number": line_number,
                    "code": "invalid_jsonl_line",
                    "message": f"{exc.msg} at column {exc.colno}",
                    "raw_line": summarize_value(stripped),
                }
            )
    return rows, errors


def load_input_records(path: Path) -> tuple[list[tuple[int, Any]], list[dict[str, Any]], str]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return [], [], "empty"
    if stripped[0] == "[":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("top-level JSON payload must be an array")
        return [(idx + 1, item) for idx, item in enumerate(payload)], [], "json_array"
    if path.suffix.lower() == ".jsonl":
        rows, errors = load_jsonl_records(text)
        return rows, errors, "jsonl"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows, errors = load_jsonl_records(text)
        return rows, errors, "jsonl"
    if isinstance(payload, dict):
        return [(1, payload)], [], "json_object"
    if isinstance(payload, list):
        return [(idx + 1, item) for idx, item in enumerate(payload)], [], "json_array"
    raise ValueError("input must be JSONL, a JSON array, or a single JSON object")


def iter_json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in JSON_SUFFIXES
    )


def load_hidden_outcomes_csv(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    warnings: list[dict[str, Any]] = []
    outcomes: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("hidden outcomes CSV is missing a header row")
        fieldnames = {name.strip(): name for name in reader.fieldnames if name is not None}
        if "event_id" not in fieldnames:
            raise ValueError("hidden outcomes CSV must include an event_id column")
        car_field = first_available_field(fieldnames, ("CAR_1_5", "hidden_CAR_1_5"))
        valence_field = first_available_field(
            fieldnames,
            ("hidden_valence", "hidden_valence_joined"),
        )
        if car_field is None:
            warnings.append(
                {
                    "scope": "hidden_outcomes",
                    "code": "missing_car_1_5_column",
                    "accepted_columns": ["CAR_1_5", "hidden_CAR_1_5"],
                }
            )
        if valence_field is None:
            warnings.append(
                {
                    "scope": "hidden_outcomes",
                    "code": "missing_hidden_valence_column",
                    "accepted_columns": ["hidden_valence", "hidden_valence_joined"],
                }
            )
        for row_number, raw_row in enumerate(reader, start=2):
            event_id = normalize_string(raw_row.get(fieldnames["event_id"]))
            if event_id is None:
                warnings.append(
                    {
                        "scope": "hidden_outcomes",
                        "code": "missing_event_id",
                        "row_number": row_number,
                    }
                )
                continue
            car_value = None
            if car_field is not None:
                car_value = coerce_float(raw_row.get(car_field))
            hidden_valence = None
            if valence_field is not None:
                hidden_valence = normalize_string(raw_row.get(valence_field))
            if event_id in outcomes:
                warnings.append(
                    {
                        "scope": "hidden_outcomes",
                        "code": "duplicate_event_id",
                        "event_id": event_id,
                        "row_number": row_number,
                    }
                )
                continue
            outcomes[event_id] = {
                "event_id": event_id,
                "CAR_1_5": car_value,
                "hidden_valence": hidden_valence,
                "row_number": row_number,
            }
    if outcomes and not any(row["CAR_1_5"] is not None for row in outcomes.values()):
        warnings.append(
            {
                "scope": "hidden_outcomes",
                "code": "no_non_null_car_1_5_values",
                "event_count": len(outcomes),
                "path": str(path),
            }
        )
    return outcomes, warnings


def load_event_metadata_csv(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    warnings: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("event metadata CSV is missing a header row")
        fieldnames = {name.strip(): name for name in reader.fieldnames if name is not None}
        if "event_id" not in fieldnames:
            raise ValueError("event metadata CSV must include an event_id column")
        available_metadata_fields = [
            field_name for field_name in EVENT_METADATA_FIELDS if field_name in fieldnames
        ]
        if "market_cap_group" not in available_metadata_fields:
            warnings.append(
                {
                    "scope": "event_metadata",
                    "code": "missing_market_cap_group_column",
                    "path": str(path),
                }
            )
        for row_number, raw_row in enumerate(reader, start=2):
            event_id = normalize_string(raw_row.get(fieldnames["event_id"]))
            if event_id is None:
                warnings.append(
                    {
                        "scope": "event_metadata",
                        "code": "missing_event_id",
                        "row_number": row_number,
                    }
                )
                continue
            if event_id in metadata:
                warnings.append(
                    {
                        "scope": "event_metadata",
                        "code": "duplicate_event_id",
                        "event_id": event_id,
                        "row_number": row_number,
                    }
                )
                continue
            metadata[event_id] = {
                field_name: normalize_string(raw_row.get(fieldnames[field_name]))
                for field_name in available_metadata_fields
            }
            metadata[event_id]["event_metadata_joined"] = True
    return metadata, warnings


def first_available_field(
    fieldnames: dict[str, str],
    candidates: tuple[str, ...],
) -> str | None:
    for candidate in candidates:
        if candidate in fieldnames:
            return fieldnames[candidate]
    return None


def load_evidence_bank_index(
    path: Path,
) -> tuple[dict[str, dict[str, set[str]]], list[dict[str, Any]], int]:
    index: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    warnings: list[dict[str, Any]] = []
    files = iter_json_files(path)
    for file_path in files:
        try:
            rows, load_errors, _ = load_input_records(file_path)
        except Exception as exc:  # pragma: no cover - defensive reporting path
            warnings.append(
                {
                    "scope": "evidence_banks",
                    "code": "unreadable_file",
                    "path": str(file_path),
                    "message": str(exc),
                }
            )
            continue
        for error in load_errors:
            warnings.append(
                {
                    "scope": "evidence_banks",
                    "code": error["code"],
                    "path": str(file_path),
                    "line_number": error["line_number"],
                    "message": error["message"],
                }
            )
        for row_number, record in rows:
            if not isinstance(record, dict):
                continue
            event_id = normalize_string(record.get("event_id"))
            evidence_units = record.get("evidence_units")
            if event_id is None or not isinstance(evidence_units, list):
                continue
            for unit in evidence_units:
                if not isinstance(unit, dict):
                    continue
                category = normalize_string(unit.get("category") or unit.get("evidence_category"))
                source_ids = normalize_string_list(unit.get("source_ids"))
                if category is None or not source_ids:
                    continue
                for source_id in source_ids:
                    index[event_id][source_id].add(category)
    return index, warnings, len(files)


def load_source_packet_index(
    path: Path,
) -> tuple[dict[str, dict[str, set[str]]], list[dict[str, Any]], int]:
    index: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    warnings: list[dict[str, Any]] = []
    files = iter_json_files(path)
    for file_path in files:
        try:
            rows, load_errors, _ = load_input_records(file_path)
        except Exception as exc:  # pragma: no cover - defensive reporting path
            warnings.append(
                {
                    "scope": "source_packets",
                    "code": "unreadable_file",
                    "path": str(file_path),
                    "message": str(exc),
                }
            )
            continue
        for error in load_errors:
            warnings.append(
                {
                    "scope": "source_packets",
                    "code": error["code"],
                    "path": str(file_path),
                    "line_number": error["line_number"],
                    "message": error["message"],
                }
            )
        for row_number, record in rows:
            if not isinstance(record, dict):
                continue
            event_id = normalize_string(record.get("event_id"))
            if event_id is None:
                continue
            source_units = record.get("source_units")
            if isinstance(source_units, list):
                for unit in source_units:
                    if not isinstance(unit, dict):
                        continue
                    source_id = normalize_string(unit.get("source_id"))
                    section = normalize_string(unit.get("section"))
                    if source_id and section:
                        index[event_id][source_id].add(f"section:{section}")
            xbrl_facts = record.get("xbrl_facts")
            if isinstance(xbrl_facts, list):
                for fact in xbrl_facts:
                    if not isinstance(fact, dict):
                        continue
                    source_id = normalize_string(fact.get("source_id"))
                    tag = normalize_string(fact.get("tag"))
                    if source_id and tag:
                        index[event_id][source_id].add(f"xbrl:{tag}")
    return index, warnings, len(files)


class TextDistanceEngine:
    def __init__(
        self,
        method_name: str,
        model_name: str | None = None,
        model: Any | None = None,
    ) -> None:
        self.method_name = method_name
        self.model_name = model_name
        self._model = model

    @classmethod
    def build(
        cls,
        method: str,
        model_name: str | None,
        warnings: list[dict[str, Any]],
    ) -> "TextDistanceEngine":
        if method == "token_cosine_proxy":
            return cls("token_char_ngram_cosine_distance_proxy")
        if method == "sentence_transformer":
            return cls._load_sentence_transformer(model_name)
        if model_name:
            try:
                return cls._load_sentence_transformer(model_name)
            except Exception as exc:
                warnings.append(
                    {
                        "scope": "rationale_distance",
                        "code": "embedding_backend_unavailable",
                        "message": str(exc),
                        "fallback_method": "token_char_ngram_cosine_distance_proxy",
                    }
                )
        return cls("token_char_ngram_cosine_distance_proxy")

    @classmethod
    def _load_sentence_transformer(cls, model_name: str | None) -> "TextDistanceEngine":
        if not model_name:
            raise ValueError(
                "--embedding-model is required when --text-distance-method=sentence_transformer"
            )
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "sentence-transformers is not installed in this environment"
            ) from exc
        model = SentenceTransformer(model_name)
        return cls("sentence_transformer_normalized_cosine_distance", model_name, model)

    def mean_pairwise_distance(self, texts: list[str]) -> float | None:
        if len(texts) < 2:
            return None
        if self._model is None:
            vectors = [lexical_feature_counter(text) for text in texts]
            return mean_pairwise_distance(vectors, cosine_distance_from_counters)
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return mean_pairwise_distance(list(embeddings), normalized_cosine_distance)


def lexical_feature_counter(text: str) -> Counter[str]:
    normalized = normalize_text(text)
    counter: Counter[str] = Counter()
    tokens = [token for token in normalized.split() if token]
    for token in tokens:
        counter[f"w:{token}"] += 1
    compact = normalized.replace(" ", "_")
    if compact:
        if len(compact) < 3:
            counter[f"c:{compact}"] += 1
        else:
            for idx in range(len(compact) - 2):
                counter[f"c:{compact[idx:idx + 3]}"] += 1
    return counter


def normalize_text(text: str) -> str:
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(cleaned.split())


def cosine_distance_from_counters(left: Counter[str], right: Counter[str]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    dot = sum(left[key] * right.get(key, 0.0) for key in left)
    similarity = max(0.0, min(1.0, dot / (left_norm * right_norm)))
    return 1.0 - similarity


def normalized_cosine_distance(left: Any, right: Any) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        dot += float(left_value) * float(right_value)
        left_norm += float(left_value) * float(left_value)
        right_norm += float(right_value) * float(right_value)
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    cosine = dot / math.sqrt(left_norm * right_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return (1.0 - cosine) / 2.0


def mean_pairwise_distance(items: list[Any], distance_fn: Any) -> float | None:
    if len(items) < 2:
        return None
    distances = [distance_fn(left, right) for left, right in combinations(items, 2)]
    return mean(distances)


def normalize_decision_record(
    row_number: int,
    record: Any,
    warnings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        warnings.append(
            {
                "scope": "decisions",
                "code": "invalid_row_type",
                "row_number": row_number,
                "received": summarize_value(record),
            }
        )
        return None

    event_id = normalize_string(record.get("event_id"))
    treatment = normalize_string(record.get("treatment"))
    profile = normalize_string(record.get("profile"))
    profile_group = normalize_string(record.get("profile_group"))
    model_family = normalize_string(record.get("model_family"))
    representation_seed = coerce_int(record.get("representation_seed"))
    decision_seed = coerce_int(record.get("decision_seed"))
    expected_return_5d = coerce_float(record.get("expected_return_5d"))
    confidence = coerce_float(record.get("confidence"))
    action = normalize_string(record.get("action"))
    action_strength = coerce_float(record.get("action_strength"))
    key_reasons = normalize_string_list(record.get("key_reasons"))
    evidence_used = normalize_string_list(record.get("evidence_used"))
    uncertainty_notes = normalize_string_list(record.get("uncertainty_notes")) or []

    required_missing = [
        field_name
        for field_name, value in (
            ("event_id", event_id),
            ("treatment", treatment),
            ("profile", profile),
            ("model_family", model_family),
            ("representation_seed", representation_seed),
            ("decision_seed", decision_seed),
            ("expected_return_5d", expected_return_5d),
            ("confidence", confidence),
            ("action", action),
            ("action_strength", action_strength),
            ("key_reasons", key_reasons),
            ("evidence_used", evidence_used),
        )
        if value is None
    ]
    if required_missing:
        warnings.append(
            {
                "scope": "decisions",
                "code": "missing_or_unparseable_fields",
                "row_number": row_number,
                "fields": required_missing,
                "event_id": event_id,
            }
        )
        return None

    assert (
        event_id is not None
        and treatment is not None
        and profile is not None
        and model_family is not None
        and representation_seed is not None
        and decision_seed is not None
        and expected_return_5d is not None
        and confidence is not None
        and action is not None
        and action_strength is not None
        and key_reasons is not None
        and evidence_used is not None
    )

    action_key = action.lower()
    if action_key not in ACTION_CODES:
        warnings.append(
            {
                "scope": "decisions",
                "code": "invalid_action",
                "row_number": row_number,
                "event_id": event_id,
                "received": action,
            }
        )
        return None

    inferred_profile_group = PROFILE_GROUP_BY_PROFILE.get(profile)
    if profile_group is None:
        profile_group = inferred_profile_group
    if profile_group is None:
        warnings.append(
            {
                "scope": "decisions",
                "code": "unknown_profile_group",
                "row_number": row_number,
                "event_id": event_id,
                "profile": profile,
            }
        )
        return None

    treatment_family = infer_treatment_family(treatment)
    upstream_model_family = upstream_model_family_from_record(record, treatment_family)
    evidence_categories_direct = normalize_string_list(record.get("evidence_categories"))
    rationale_text = " ".join(item.strip() for item in key_reasons if item.strip())

    return {
        "row_number": row_number,
        "event_id": event_id,
        "treatment": treatment,
        "treatment_family": treatment_family,
        "profile": profile,
        "profile_canonical": CANONICAL_PROFILE_BY_PROFILE.get(profile, profile),
        "profile_group": profile_group,
        "model_family": model_family,
        "upstream_model_family": upstream_model_family,
        "representation_seed": representation_seed,
        "decision_seed": decision_seed,
        "expected_return_5d": expected_return_5d,
        "confidence": confidence,
        "action": action_key,
        "action_code": ACTION_CODES[action_key],
        "action_strength": action_strength,
        "key_reasons": key_reasons,
        "rationale_text": rationale_text,
        "evidence_used": evidence_used,
        "uncertainty_notes": uncertainty_notes,
        "evidence_categories_direct": evidence_categories_direct,
    }


def attach_evidence_categories(
    decisions: list[dict[str, Any]],
    category_index: dict[str, dict[str, set[str]]] | None,
    category_source_name: str | None,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for decision in decisions:
        row = dict(decision)
        direct_categories = row.get("evidence_categories_direct")
        if direct_categories:
            row["evidence_categories"] = sorted(set(direct_categories))
            row["evidence_category_source"] = "decision_record"
            row["evidence_category_coverage"] = 1.0
            enriched.append(row)
            continue

        if category_index is None or category_source_name is None:
            row["evidence_categories"] = None
            row["evidence_category_source"] = "unavailable"
            row["evidence_category_coverage"] = None
            enriched.append(row)
            continue

        event_mapping = category_index.get(row["event_id"], {})
        categories: set[str] = set()
        mapped_source_ids = 0
        evidence_ids = set(row["evidence_used"])
        for source_id in evidence_ids:
            source_categories = event_mapping.get(source_id)
            if source_categories:
                mapped_source_ids += 1
                categories.update(source_categories)
        coverage = (
            mapped_source_ids / len(evidence_ids)
            if evidence_ids
            else None
        )
        row["evidence_categories"] = sorted(categories)
        row["evidence_category_source"] = category_source_name
        row["evidence_category_coverage"] = coverage
        enriched.append(row)
    return enriched


def attach_hidden_outcomes(
    decisions: list[dict[str, Any]],
    hidden_outcomes: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if hidden_outcomes is None:
        return [with_quality_fields(decision, None) for decision in decisions]
    return [
        with_quality_fields(decision, hidden_outcomes.get(decision["event_id"]))
        for decision in decisions
    ]


def attach_event_metadata(
    rows: list[dict[str, Any]],
    event_metadata: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        metadata = None if event_metadata is None else event_metadata.get(row["event_id"])
        enriched_row = dict(row)
        for field_name in EVENT_METADATA_FIELDS:
            enriched_row[field_name] = None if metadata is None else metadata.get(field_name)
        enriched_row["event_metadata_joined"] = metadata is not None
        enriched.append(enriched_row)
    return enriched


def with_quality_fields(
    decision: dict[str, Any],
    hidden_outcome: dict[str, Any] | None,
) -> dict[str, Any]:
    row = dict(decision)
    if hidden_outcome is None:
        row["CAR_1_5"] = None
        row["hidden_valence"] = None
        row["directional_accuracy"] = None
        row["action_accuracy"] = None
        row["signed_return_error"] = None
        row["absolute_return_error"] = None
        return row

    car_1_5 = hidden_outcome.get("CAR_1_5")
    row["CAR_1_5"] = car_1_5
    row["hidden_valence"] = hidden_outcome.get("hidden_valence")
    if car_1_5 is None:
        row["directional_accuracy"] = None
        row["action_accuracy"] = None
        row["signed_return_error"] = None
        row["absolute_return_error"] = None
        return row

    outcome_sign = sign(car_1_5)
    row["directional_accuracy"] = (
        1.0 if sign(row["expected_return_5d"]) == outcome_sign else 0.0
    )
    row["action_accuracy"] = 1.0 if row["action_code"] == outcome_sign else 0.0
    signed_error = row["expected_return_5d"] - car_1_5
    row["signed_return_error"] = signed_error
    row["absolute_return_error"] = abs(signed_error)
    return row


def sign(value: float) -> int:
    if value > SIGN_EPSILON:
        return 1
    if value < -SIGN_EPSILON:
        return -1
    return 0


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def population_std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    center = sum(values) / len(values)
    variance = sum((value - center) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def quantile_linear(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    if lower_index == upper_index:
        return lower_value
    weight = position - lower_index
    return lower_value + weight * (upper_value - lower_value)


def interquartile_range(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return quantile_linear(ordered, 0.75) - quantile_linear(ordered, 0.25)


def mean_pairwise_abs_diff(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    diffs = [abs(left - right) for left, right in combinations(values, 2)]
    return mean(diffs)


def entropy_base2(values: list[str]) -> float | None:
    if not values:
        return None
    counts = Counter(values)
    total = len(values)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def hhi(values: list[str], categories: tuple[str, ...]) -> float | None:
    if not values:
        return None
    counts = Counter(values)
    total = len(values)
    return sum((counts.get(category, 0) / total) ** 2 for category in categories)


def mean_pairwise_jaccard(sets: list[set[str]]) -> float | None:
    if len(sets) < 2:
        return None
    scores: list[float] = []
    for left, right in combinations(sets, 2):
        union = left | right
        if not union:
            scores.append(1.0)
        else:
            scores.append(len(left & right) / len(union))
    return mean(scores)


def group_decisions_by_cell(
    decisions: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str, int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for decision in decisions:
        key = (
            decision["event_id"],
            decision["treatment"],
            decision["upstream_model_family"],
            decision["model_family"],
            decision["representation_seed"],
            decision["decision_seed"],
        )
        grouped[key].append(decision)
    return grouped


def select_investor_group_rows(
    cell_rows: list[dict[str, Any]],
    investor_group: str,
) -> list[dict[str, Any]]:
    if investor_group == "all_profiles":
        return list(cell_rows)
    return [row for row in cell_rows if row["profile_group"] == investor_group]


def compute_cell_metrics(
    cell_rows: list[dict[str, Any]],
    investor_group: str,
    text_distance_engine: TextDistanceEngine,
) -> dict[str, Any]:
    base = cell_rows[0]
    subset = select_investor_group_rows(cell_rows, investor_group)
    profiles = sorted(row["profile_canonical"] for row in subset)
    expected_profiles = EXPECTED_PROFILES_BY_INVESTOR_GROUP[investor_group]
    missing_profiles = sorted(set(expected_profiles) - set(profiles))
    pair_count = len(subset) * (len(subset) - 1) // 2

    expected_returns = [row["expected_return_5d"] for row in subset]
    confidences = [row["confidence"] for row in subset]
    actions = [row["action"] for row in subset]
    action_codes = [row["action_code"] for row in subset]
    rationale_texts = [row["rationale_text"] for row in subset]
    evidence_sets = [set(row["evidence_used"]) for row in subset]

    all_have_category_sets = all(row["evidence_categories"] is not None for row in subset)
    evidence_category_sets = (
        [set(row["evidence_categories"]) for row in subset]
        if all_have_category_sets
        else None
    )
    mean_category_coverage = mean(
        row["evidence_category_coverage"]
        for row in subset
        if row["evidence_category_coverage"] is not None
    )
    category_sources = sorted({row["evidence_category_source"] for row in subset})
    car_values = [row["CAR_1_5"] for row in subset if row["CAR_1_5"] is not None]
    directional_accuracy_values = [
        row["directional_accuracy"]
        for row in subset
        if row["directional_accuracy"] is not None
    ]
    action_accuracy_values = [
        row["action_accuracy"]
        for row in subset
        if row["action_accuracy"] is not None
    ]
    absolute_return_errors = [
        row["absolute_return_error"]
        for row in subset
        if row["absolute_return_error"] is not None
    ]
    signed_return_errors = [
        row["signed_return_error"]
        for row in subset
        if row["signed_return_error"] is not None
    ]

    hhi_value = hhi(actions, tuple(ACTION_CODES))
    absolute_net_direction = abs(sum(action_codes) / len(action_codes)) if action_codes else None
    source_overlap = mean_pairwise_jaccard(evidence_sets)
    if evidence_category_sets is None or mean_category_coverage in {None, 0.0}:
        evidence_category_overlap = None
    else:
        evidence_category_overlap = mean_pairwise_jaccard(evidence_category_sets)

    return {
        "event_id": base["event_id"],
        **{field_name: base.get(field_name) for field_name in EVENT_METADATA_FIELDS},
        "event_metadata_joined": base.get("event_metadata_joined", False),
        "treatment": base["treatment"],
        "treatment_family": base["treatment_family"],
        "upstream_model_family": base["upstream_model_family"],
        "model_family": base["model_family"],
        "investor_group": investor_group,
        "representation_seed": base["representation_seed"],
        "decision_seed": base["decision_seed"],
        "profile_count": len(subset),
        "expected_profile_count": len(expected_profiles),
        "is_complete_group": len(subset) == len(expected_profiles) and not missing_profiles,
        "profiles_json": json.dumps(profiles, separators=(",", ":")),
        "missing_profiles_json": json.dumps(missing_profiles, separators=(",", ":")),
        "pair_count": pair_count,
        "rationale_distance_method": text_distance_engine.method_name,
        "evidence_category_source": (
            category_sources[0] if len(category_sources) == 1 else "mixed"
        ),
        "mean_evidence_category_coverage": mean_category_coverage,
        "hidden_outcomes_available": bool(car_values),
        "CAR_1_5": car_values[0] if car_values else None,
        "hidden_valence": base.get("hidden_valence"),
        "std_expected_return_5d": population_std(expected_returns),
        "mean_pairwise_abs_diff_expected_return_5d": mean_pairwise_abs_diff(expected_returns),
        "iqr_expected_return_5d": interquartile_range(expected_returns),
        "confidence_dispersion": population_std(confidences),
        "entropy_buy_hold_sell": entropy_base2(actions),
        "hhi_buy_hold_sell": hhi_value,
        "action_diversity_one_minus_hhi": (
            None if hhi_value is None else 1.0 - hhi_value
        ),
        "absolute_net_direction": absolute_net_direction,
        "action_direction_diversity_one_minus_abs_net_direction": (
            None if absolute_net_direction is None else 1.0 - absolute_net_direction
        ),
        "mean_pairwise_sentence_embedding_distance": text_distance_engine.mean_pairwise_distance(
            rationale_texts
        ),
        "source_id_overlap": source_overlap,
        "source_id_diversity_one_minus_overlap": (
            None if source_overlap is None else 1.0 - source_overlap
        ),
        "evidence_category_overlap": evidence_category_overlap,
        "evidence_category_diversity_one_minus_overlap": (
            None
            if evidence_category_overlap is None
            else 1.0 - evidence_category_overlap
        ),
        "quality_directional_accuracy": mean(directional_accuracy_values),
        "quality_action_accuracy": mean(action_accuracy_values),
        "quality_mean_absolute_return_error": mean(absolute_return_errors),
        "quality_mean_signed_return_error": mean(signed_return_errors),
    }


def build_cell_metric_rows(
    decisions: list[dict[str, Any]],
    text_distance_engine: TextDistanceEngine,
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cell_rows: list[dict[str, Any]] = []
    for key, grouped_rows in sorted(group_decisions_by_cell(decisions).items()):
        duplicate_profiles = sorted(
            profile
            for profile, count in Counter(
                row["profile_canonical"] for row in grouped_rows
            ).items()
            if count > 1
        )
        if duplicate_profiles:
            warnings.append(
                {
                    "scope": "cell_metrics",
                    "code": "duplicate_profiles_in_cell",
                    "event_id": key[0],
                    "treatment": key[1],
                    "upstream_model_family": key[2],
                    "model_family": key[3],
                    "representation_seed": key[4],
                    "decision_seed": key[5],
                    "profiles": duplicate_profiles,
                }
            )
            continue
        for investor_group in INVESTOR_GROUPS:
            subset = select_investor_group_rows(grouped_rows, investor_group)
            if not subset:
                continue
            cell_rows.append(
                compute_cell_metrics(grouped_rows, investor_group, text_distance_engine)
            )
    return cell_rows


def build_t2_t3_contrasts(
    cell_metric_rows: list[dict[str, Any]],
    sci_near_zero_threshold: float,
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cell_metric_rows:
        key = (
            row["event_id"],
            row["upstream_model_family"],
            row["model_family"],
            row["investor_group"],
            row["representation_seed"],
            row["decision_seed"],
        )
        grouped[key].append(row)

    contrast_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        family_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            family = row.get("treatment_family")
            if family:
                family_buckets[family].append(row)
        t2_rows = family_buckets.get("T2", [])
        t3_rows = family_buckets.get("T3", [])
        if not t2_rows or not t3_rows:
            continue
        if len(t2_rows) != 1 or len(t3_rows) != 1:
            warnings.append(
                {
                    "scope": "contrasts",
                    "code": "ambiguous_t2_t3_pair",
                    "event_id": key[0],
                    "upstream_model_family": key[1],
                    "model_family": key[2],
                    "investor_group": key[3],
                    "representation_seed": key[4],
                    "decision_seed": key[5],
                    "t2_rows": len(t2_rows),
                    "t3_rows": len(t3_rows),
                }
            )
            continue

        t2_row = t2_rows[0]
        t3_row = t3_rows[0]
        contrast_row: dict[str, Any] = {
            "event_id": key[0],
            **{field_name: t2_row.get(field_name) or t3_row.get(field_name) for field_name in EVENT_METADATA_FIELDS},
            "event_metadata_joined": (
                bool(t2_row.get("event_metadata_joined"))
                or bool(t3_row.get("event_metadata_joined"))
            ),
            "upstream_model_family": key[1],
            "model_family": key[2],
            "investor_group": key[3],
            "representation_seed": key[4],
            "decision_seed": key[5],
            "t2_treatment": t2_row["treatment"],
            "t3_treatment": t3_row["treatment"],
            "t2_profile_count": t2_row["profile_count"],
            "t3_profile_count": t3_row["profile_count"],
            "t2_is_complete_group": t2_row["is_complete_group"],
            "t3_is_complete_group": t3_row["is_complete_group"],
            "rationale_distance_method": t2_row["rationale_distance_method"],
            "CAR_1_5": (
                t2_row["CAR_1_5"] if t2_row["CAR_1_5"] is not None else t3_row["CAR_1_5"]
            ),
            "hidden_valence": t2_row["hidden_valence"] or t3_row["hidden_valence"],
        }
        for metric_name in CONTRAST_METRIC_NAMES:
            t2_value = t2_row.get(metric_name)
            t3_value = t3_row.get(metric_name)
            contrast_row[f"t2_{metric_name}"] = t2_value
            contrast_row[f"t3_{metric_name}"] = t3_value
            if t2_value is None or t3_value is None:
                contrast_row[f"delta_{metric_name}"] = None
                contrast_row[f"sci_{metric_name}"] = None
                contrast_row[f"sci_{metric_name}_stable"] = False
                continue
            delta = t3_value - t2_value
            contrast_row[f"delta_{metric_name}"] = delta
            if t3_value < sci_near_zero_threshold:
                contrast_row[f"sci_{metric_name}"] = None
                contrast_row[f"sci_{metric_name}_stable"] = False
            else:
                contrast_row[f"sci_{metric_name}"] = delta / t3_value
                contrast_row[f"sci_{metric_name}_stable"] = True
        contrast_rows.append(contrast_row)
    return contrast_rows


def treatment_row_key(row: dict[str, Any]) -> tuple[str, str, str, str, int, int, str]:
    return (
        row["event_id"],
        row["upstream_model_family"],
        row["model_family"],
        row["investor_group"],
        row["representation_seed"],
        row["decision_seed"],
        row["treatment_family"],
    )


def one_or_none(
    rows: list[dict[str, Any]],
    *,
    warnings: list[dict[str, Any]],
    scope: str,
    key: tuple[Any, ...],
) -> dict[str, Any] | None:
    if len(rows) == 1:
        return rows[0]
    if len(rows) > 1:
        warnings.append(
            {
                "scope": scope,
                "code": "ambiguous_treatment_cell",
                "key": list(key),
                "row_count": len(rows),
            }
        )
    return None


def add_treatment_contrast_rows(
    *,
    contrast_rows: list[dict[str, Any]],
    contrast_id: str,
    baseline_row: dict[str, Any],
    comparison_row: dict[str, Any],
    denominator_row: dict[str, Any] | None,
    index_type: str,
    sci_near_zero_threshold: float,
) -> None:
    upstream_model_family = comparison_row["upstream_model_family"]
    if baseline_row["treatment_family"] not in {"T1", "B0"}:
        upstream_model_family = baseline_row["upstream_model_family"]
    for metric_name in CONTRAST_METRIC_NAMES:
        baseline_value = baseline_row.get(metric_name)
        comparison_value = comparison_row.get(metric_name)
        denominator_value = (
            None if denominator_row is None else denominator_row.get(metric_name)
        )
        delta = (
            None
            if baseline_value is None or comparison_value is None
            else comparison_value - baseline_value
        )
        normalized_index = None
        is_stable = False
        if delta is not None and index_type in {"SCI", "SEMI"}:
            if (
                denominator_value is not None
                and abs(denominator_value) >= sci_near_zero_threshold
            ):
                normalized_index = delta / denominator_value
                is_stable = True
        elif delta is not None:
            is_stable = True

        contrast_rows.append(
            {
                "event_id": baseline_row["event_id"],
                **{
                    field_name: baseline_row.get(field_name) or comparison_row.get(field_name)
                    for field_name in EVENT_METADATA_FIELDS
                },
                "event_metadata_joined": (
                    bool(baseline_row.get("event_metadata_joined"))
                    or bool(comparison_row.get("event_metadata_joined"))
                ),
                "upstream_model_family": upstream_model_family,
                "model_family": baseline_row["model_family"],
                "investor_group": baseline_row["investor_group"],
                "representation_seed": comparison_row["representation_seed"],
                "decision_seed": baseline_row["decision_seed"],
                "contrast_id": contrast_id,
                "baseline_treatment_family": baseline_row["treatment_family"],
                "comparison_treatment_family": comparison_row["treatment_family"],
                "baseline_treatment": baseline_row["treatment"],
                "comparison_treatment": comparison_row["treatment"],
                "baseline_profile_count": baseline_row["profile_count"],
                "comparison_profile_count": comparison_row["profile_count"],
                "baseline_is_complete_group": baseline_row["is_complete_group"],
                "comparison_is_complete_group": comparison_row["is_complete_group"],
                "denominator_treatment_family": (
                    None if denominator_row is None else denominator_row["treatment_family"]
                ),
                "denominator_treatment": (
                    None if denominator_row is None else denominator_row["treatment"]
                ),
                "metric_name": metric_name,
                "metric_group": METRIC_GROUPS[metric_name],
                "metric_unit": METRIC_UNITS[metric_name],
                "baseline_value": baseline_value,
                "comparison_value": comparison_value,
                "denominator_value": denominator_value,
                "delta": delta,
                "normalized_index": normalized_index,
                "index_type": index_type,
                "is_stable": is_stable,
                "epsilon": sci_near_zero_threshold,
                "rationale_distance_method": comparison_row["rationale_distance_method"],
                "CAR_1_5": (
                    baseline_row["CAR_1_5"]
                    if baseline_row["CAR_1_5"] is not None
                    else comparison_row["CAR_1_5"]
                ),
                "hidden_valence": (
                    baseline_row["hidden_valence"] or comparison_row["hidden_valence"]
                ),
            }
        )


def build_treatment_contrasts(
    cell_metric_rows: list[dict[str, Any]],
    sci_near_zero_threshold: float,
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_full_key: dict[tuple[str, str, str, str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    by_event_model_group_decision: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cell_metric_rows:
        by_full_key[treatment_row_key(row)].append(row)
        by_event_model_group_decision[
            (
                row["event_id"],
                row["model_family"],
                row["investor_group"],
                row["decision_seed"],
            )
        ].append(row)

    contrast_rows: list[dict[str, Any]] = []

    for full_key, rows in sorted(by_full_key.items()):
        (
            event_id,
            upstream_model_family,
            model_family,
            investor_group,
            representation_seed,
            decision_seed,
            family,
        ) = full_key
        if family != "T2":
            continue
        t2_row = one_or_none(
            rows,
            warnings=warnings,
            scope="treatment_contrasts",
            key=full_key,
        )
        if t2_row is None:
            continue
        shared_key = (
            event_id,
            upstream_model_family,
            model_family,
            investor_group,
            representation_seed,
            decision_seed,
        )
        t3_row = one_or_none(
            by_full_key.get((*shared_key, "T3"), []),
            warnings=warnings,
            scope="treatment_contrasts",
            key=(*shared_key, "T3"),
        )
        t4_row = one_or_none(
            by_full_key.get((*shared_key, "T4"), []),
            warnings=warnings,
            scope="treatment_contrasts",
            key=(*shared_key, "T4"),
        )
        if t3_row is not None:
            add_treatment_contrast_rows(
                contrast_rows=contrast_rows,
                contrast_id="T3_minus_T2",
                baseline_row=t2_row,
                comparison_row=t3_row,
                denominator_row=t3_row,
                index_type="SCI",
                sci_near_zero_threshold=sci_near_zero_threshold,
            )
        if t4_row is not None:
            add_treatment_contrast_rows(
                contrast_rows=contrast_rows,
                contrast_id="T4_minus_T2",
                baseline_row=t2_row,
                comparison_row=t4_row,
                denominator_row=t3_row,
                index_type="SEMI",
                sci_near_zero_threshold=sci_near_zero_threshold,
            )

    for key, rows in sorted(by_event_model_group_decision.items()):
        t1_row = one_or_none(
            [row for row in rows if row.get("treatment_family") == "T1"],
            warnings=warnings,
            scope="treatment_contrasts",
            key=(*key, "T1"),
        )
        if t1_row is not None:
            for comparison_family in ("B0", "T2", "T3", "T4"):
                comparison_rows = [
                    row for row in rows if row.get("treatment_family") == comparison_family
                ]
                for comparison_row in sorted(
                    comparison_rows,
                    key=lambda item: (item["representation_seed"], item["treatment"]),
                ):
                    add_treatment_contrast_rows(
                        contrast_rows=contrast_rows,
                        contrast_id=f"{comparison_family}_minus_T1",
                        baseline_row=t1_row,
                        comparison_row=comparison_row,
                        denominator_row=None,
                        index_type="raw_delta",
                        sci_near_zero_threshold=sci_near_zero_threshold,
                    )

        b0_row = one_or_none(
            [row for row in rows if row.get("treatment_family") == "B0"],
            warnings=warnings,
            scope="treatment_contrasts",
            key=(*key, "B0"),
        )
        if b0_row is None:
            continue
        for comparison_family in ("T2", "T3", "T4"):
            comparison_rows = [
                row for row in rows if row.get("treatment_family") == comparison_family
            ]
            for comparison_row in sorted(
                comparison_rows,
                key=lambda item: (
                    item["upstream_model_family"],
                    item["representation_seed"],
                    item["treatment"],
                ),
            ):
                add_treatment_contrast_rows(
                    contrast_rows=contrast_rows,
                    contrast_id=f"{comparison_family}_minus_B0",
                    baseline_row=b0_row,
                    comparison_row=comparison_row,
                    denominator_row=None,
                    index_type="raw_delta",
                    sci_near_zero_threshold=sci_near_zero_threshold,
                )

    return contrast_rows


def serialize_csv_value(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, set):
        value = sorted(value)
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    fieldname: serialize_csv_value(row.get(fieldname))
                    for fieldname in fieldnames
                }
            )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def decision_row_fieldnames() -> list[str]:
    return [
        "row_number",
        "event_id",
        *EVENT_METADATA_FIELDS,
        "event_metadata_joined",
        "treatment",
        "treatment_family",
        "profile",
        "profile_canonical",
        "profile_group",
        "upstream_model_family",
        "model_family",
        "representation_seed",
        "decision_seed",
        "expected_return_5d",
        "confidence",
        "action",
        "action_code",
        "action_strength",
        "rationale_text",
        "key_reasons",
        "evidence_used",
        "evidence_categories",
        "evidence_category_source",
        "evidence_category_coverage",
        "uncertainty_notes",
        "CAR_1_5",
        "hidden_valence",
        "directional_accuracy",
        "action_accuracy",
        "signed_return_error",
        "absolute_return_error",
    ]


def cell_metric_fieldnames() -> list[str]:
    base_fields = [
        "event_id",
        *EVENT_METADATA_FIELDS,
        "event_metadata_joined",
        "treatment",
        "treatment_family",
        "upstream_model_family",
        "model_family",
        "investor_group",
        "representation_seed",
        "decision_seed",
        "profile_count",
        "expected_profile_count",
        "is_complete_group",
        "profiles_json",
        "missing_profiles_json",
        "pair_count",
        "rationale_distance_method",
        "evidence_category_source",
        "mean_evidence_category_coverage",
        "hidden_outcomes_available",
        "CAR_1_5",
        "hidden_valence",
    ]
    quality_fields = [
        "quality_directional_accuracy",
        "quality_action_accuracy",
        "quality_mean_absolute_return_error",
        "quality_mean_signed_return_error",
    ]
    return base_fields + list(CELL_METRIC_NAMES) + quality_fields


def contrast_fieldnames() -> list[str]:
    base_fields = [
        "event_id",
        *EVENT_METADATA_FIELDS,
        "event_metadata_joined",
        "upstream_model_family",
        "model_family",
        "investor_group",
        "representation_seed",
        "decision_seed",
        "t2_treatment",
        "t3_treatment",
        "t2_profile_count",
        "t3_profile_count",
        "t2_is_complete_group",
        "t3_is_complete_group",
        "rationale_distance_method",
        "CAR_1_5",
        "hidden_valence",
    ]
    metric_fields: list[str] = []
    for metric_name in CONTRAST_METRIC_NAMES:
        metric_fields.extend(
            [
                f"t2_{metric_name}",
                f"t3_{metric_name}",
                f"delta_{metric_name}",
                f"sci_{metric_name}",
                f"sci_{metric_name}_stable",
            ]
        )
    return base_fields + metric_fields


def treatment_contrast_fieldnames() -> list[str]:
    return [
        "event_id",
        *EVENT_METADATA_FIELDS,
        "event_metadata_joined",
        "upstream_model_family",
        "model_family",
        "investor_group",
        "representation_seed",
        "decision_seed",
        "contrast_id",
        "baseline_treatment_family",
        "comparison_treatment_family",
        "baseline_treatment",
        "comparison_treatment",
        "baseline_profile_count",
        "comparison_profile_count",
        "baseline_is_complete_group",
        "comparison_is_complete_group",
        "denominator_treatment_family",
        "denominator_treatment",
        "metric_name",
        "metric_group",
        "metric_unit",
        "baseline_value",
        "comparison_value",
        "denominator_value",
        "delta",
        "normalized_index",
        "index_type",
        "is_stable",
        "epsilon",
        "rationale_distance_method",
        "CAR_1_5",
        "hidden_valence",
    ]


def build_run_summary(
    *,
    input_path: Path,
    input_format: str,
    hidden_outcomes_path: Path | None,
    event_metadata_path: Path | None,
    evidence_banks_path: Path | None,
    source_packets_path: Path | None,
    text_distance_engine: TextDistanceEngine,
    sci_near_zero_threshold: float,
    total_input_rows: int,
    load_errors: list[dict[str, Any]],
    usable_decisions: int,
    decision_rows: list[dict[str, Any]],
    cell_metric_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
    treatment_contrast_rows: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    evidence_bank_file_count: int,
    source_packet_file_count: int,
    hidden_outcome_event_count: int,
    event_metadata_count: int,
    output_paths: dict[str, str],
) -> dict[str, Any]:
    mapped_category_rows = sum(
        1 for row in decision_rows if row["evidence_categories"] is not None
    )
    hidden_outcome_rows = sum(1 for row in decision_rows if row["CAR_1_5"] is not None)
    return {
        "generated_at_utc": utc_now_iso(),
        "input_path": str(input_path),
        "input_format": input_format,
        "output_paths": output_paths,
        "summary": {
            "total_input_rows": total_input_rows,
            "load_errors": len(load_errors),
            "usable_decisions": usable_decisions,
            "decision_rows_written": len(decision_rows),
            "cell_metric_rows_written": len(cell_metric_rows),
            "t2_t3_contrast_rows_written": len(contrast_rows),
            "treatment_contrast_rows_written": len(treatment_contrast_rows),
            "warnings": len(warnings),
            "hidden_outcomes_joined_rows": hidden_outcome_rows,
            "hidden_outcomes_events_loaded": hidden_outcome_event_count,
            "event_metadata_events_loaded": event_metadata_count,
            "event_metadata_joined_rows": sum(
                1 for row in decision_rows if row.get("event_metadata_joined")
            ),
            "evidence_category_rows_mapped": mapped_category_rows,
            "evidence_bank_files_scanned": evidence_bank_file_count,
            "source_packet_files_scanned": source_packet_file_count,
        },
        "inputs": {
            "hidden_outcomes_csv": None if hidden_outcomes_path is None else str(hidden_outcomes_path),
            "event_metadata_csv": None if event_metadata_path is None else str(event_metadata_path),
            "evidence_banks": None if evidence_banks_path is None else str(evidence_banks_path),
            "source_packets": None if source_packets_path is None else str(source_packets_path),
        },
        "metric_units": METRIC_UNITS,
        "contrast_metric_units": {
            metric_name: METRIC_UNITS[metric_name] for metric_name in CONTRAST_METRIC_NAMES
        },
        "contrast_metric_groups": {
            metric_name: METRIC_GROUPS[metric_name] for metric_name in CONTRAST_METRIC_NAMES
        },
        "rationale_distance": {
            "method": text_distance_engine.method_name,
            "embedding_model": text_distance_engine.model_name,
            "fallback_note": (
                "When no local embedding model is supplied, the script uses "
                "cosine distance over deterministic token and character n-gram counts."
            ),
        },
        "assumptions": {
            **ASSUMPTIONS,
            "sci_near_zero_threshold": sci_near_zero_threshold,
            "evidence_category_mapping_priority": [
                "decision_record.evidence_categories",
                "evidence_banks category mapping",
                "source_packets section/tag fallback",
                "null if no mapping source is available",
            ],
        },
        "warnings": warnings,
        "load_errors_detail": load_errors,
    }


def print_summary(summary: dict[str, Any]) -> None:
    stats = summary["summary"]
    print(
        "loaded"
        f" {stats['usable_decisions']} usable decisions from {summary['input_path']}"
        f" ({summary['input_format']})"
    )
    print(
        "wrote"
        f" {stats['cell_metric_rows_written']} cell rows and"
        f" {stats['t2_t3_contrast_rows_written']} T2-vs-T3 contrast rows"
        f" ({stats['treatment_contrast_rows_written']} long contrast rows)"
    )
    print(f"outputs={summary['output_paths']['output_dir']}")
    print(f"warnings={stats['warnings']}")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"input not found: {input_path}", file=sys.stderr)
        return 1

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else default_output_dir(input_path).resolve()
    )

    try:
        rows, load_errors, input_format = load_input_records(input_path)
    except Exception as exc:
        print(f"failed to load decisions: {exc}", file=sys.stderr)
        return 1

    warnings: list[dict[str, Any]] = []
    normalized_decisions: list[dict[str, Any]] = []
    for row_number, record in rows:
        normalized = normalize_decision_record(row_number, record, warnings)
        if normalized is not None:
            normalized_decisions.append(normalized)

    hidden_outcomes_path = (
        Path(args.hidden_outcomes_csv).resolve() if args.hidden_outcomes_csv else None
    )
    hidden_outcomes = None
    hidden_outcome_event_count = 0
    if hidden_outcomes_path is not None:
        try:
            hidden_outcomes, hidden_warnings = load_hidden_outcomes_csv(hidden_outcomes_path)
        except Exception as exc:
            print(f"failed to load hidden outcomes CSV: {exc}", file=sys.stderr)
            return 1
        warnings.extend(hidden_warnings)
        hidden_outcome_event_count = len(hidden_outcomes)

    event_metadata_path = (
        Path(args.event_metadata_csv).resolve() if args.event_metadata_csv else None
    )
    event_metadata = None
    event_metadata_count = 0
    if event_metadata_path is not None:
        try:
            event_metadata, metadata_warnings = load_event_metadata_csv(event_metadata_path)
        except Exception as exc:
            print(f"failed to load event metadata CSV: {exc}", file=sys.stderr)
            return 1
        warnings.extend(metadata_warnings)
        event_metadata_count = len(event_metadata)

    evidence_banks_path = (
        Path(args.evidence_banks).resolve() if args.evidence_banks else None
    )
    source_packets_path = (
        Path(args.source_packets).resolve() if args.source_packets else None
    )

    evidence_bank_index = None
    source_packet_index = None
    evidence_bank_file_count = 0
    source_packet_file_count = 0

    if evidence_banks_path is not None:
        try:
            evidence_bank_index, bank_warnings, evidence_bank_file_count = load_evidence_bank_index(
                evidence_banks_path
            )
        except Exception as exc:
            print(f"failed to load evidence banks: {exc}", file=sys.stderr)
            return 1
        warnings.extend(bank_warnings)

    if source_packets_path is not None:
        try:
            source_packet_index, packet_warnings, source_packet_file_count = load_source_packet_index(
                source_packets_path
            )
        except Exception as exc:
            print(f"failed to load source packets: {exc}", file=sys.stderr)
            return 1
        warnings.extend(packet_warnings)

    if evidence_bank_index is not None:
        category_index = evidence_bank_index
        category_source_name = "evidence_banks"
    elif source_packet_index is not None:
        category_index = source_packet_index
        category_source_name = "source_packets_section_tag_fallback"
    else:
        category_index = None
        category_source_name = None

    text_distance_engine = TextDistanceEngine.build(
        args.text_distance_method,
        args.embedding_model,
        warnings,
    )

    decision_rows = attach_event_metadata(
        attach_hidden_outcomes(
            attach_evidence_categories(
                normalized_decisions,
                category_index,
                category_source_name,
            ),
            hidden_outcomes,
        ),
        event_metadata,
    )

    cell_metric_rows = build_cell_metric_rows(decision_rows, text_distance_engine, warnings)
    contrast_rows = build_t2_t3_contrasts(
        cell_metric_rows,
        args.sci_near_zero_threshold,
        warnings,
    )
    treatment_contrast_rows = build_treatment_contrasts(
        cell_metric_rows,
        args.sci_near_zero_threshold,
        warnings,
    )

    output_paths = {
        "output_dir": str(output_dir),
        "decision_rows_csv": str(output_dir / "decision_rows.csv"),
        "cell_metrics_csv": str(output_dir / "cell_metrics.csv"),
        "t2_t3_contrasts_csv": str(output_dir / "t2_t3_contrasts.csv"),
        "treatment_contrasts_csv": str(output_dir / "treatment_contrasts.csv"),
        "run_summary_json": str(output_dir / "run_summary.json"),
    }

    write_csv(output_dir / "decision_rows.csv", decision_rows, decision_row_fieldnames())
    write_csv(output_dir / "cell_metrics.csv", cell_metric_rows, cell_metric_fieldnames())
    write_csv(output_dir / "t2_t3_contrasts.csv", contrast_rows, contrast_fieldnames())
    write_csv(
        output_dir / "treatment_contrasts.csv",
        treatment_contrast_rows,
        treatment_contrast_fieldnames(),
    )

    run_summary = build_run_summary(
        input_path=input_path,
        input_format=input_format,
        hidden_outcomes_path=hidden_outcomes_path,
        event_metadata_path=event_metadata_path,
        evidence_banks_path=evidence_banks_path,
        source_packets_path=source_packets_path,
        text_distance_engine=text_distance_engine,
        sci_near_zero_threshold=args.sci_near_zero_threshold,
        total_input_rows=len(rows),
        load_errors=load_errors,
        usable_decisions=len(normalized_decisions),
        decision_rows=decision_rows,
        cell_metric_rows=cell_metric_rows,
        contrast_rows=contrast_rows,
        treatment_contrast_rows=treatment_contrast_rows,
        warnings=warnings,
        evidence_bank_file_count=evidence_bank_file_count,
        source_packet_file_count=source_packet_file_count,
        hidden_outcome_event_count=hidden_outcome_event_count,
        event_metadata_count=event_metadata_count,
        output_paths=output_paths,
    )
    write_json(output_dir / "run_summary.json", run_summary)
    print_summary(run_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
