#!/usr/bin/env python3
"""Compute semantic reasoning-diversity diagnostics from decision rows.

This script is intentionally separate from ``compute_diversity_metrics.py``.
The existing diversity script keeps the historical lexical rationale-distance
metric. This script adds a cleaner post-hoc reasoning analysis over three text
channels:

- expressed reasons: ``key_reasons``
- uncertainty frames: ``uncertainty_notes``
- composite reasoning: ``key_reasons + uncertainty_notes``

It does not include source IDs in the text metrics, so evidence-path diversity
remains analytically separate from expressed-reason diversity.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECISION_ROWS = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "t4_followup_20260505"
    / "metrics_stage2_plus_t4_four_models_profile_expanded_20260505"
    / "decision_rows.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "docs"
    / "experiment_results"
    / "tables"
    / "reasoning_semantic_diversity_20260507"
)
DEFAULT_MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2"
    / "snapshots/84fccfe766bcfd679e39efefe4ebf45af190ad2d"
)

TEXT_CHANNELS = (
    "expressed_reason",
    "uncertainty_frame",
    "composite_reasoning",
)
INVESTOR_GROUPS = ("all_profiles", "retail", "institutional")
CONTRASTS = (
    ("B0_minus_T1", "B0", "T1"),
    ("T2_minus_B0", "T2", "B0"),
    ("T3_minus_B0", "T3", "B0"),
    ("T4_minus_B0", "T4", "B0"),
    ("T3_minus_T2", "T3", "T2"),
    ("T4_minus_T2", "T4", "T2"),
    ("T4_minus_T3", "T4", "T3"),
    ("T4_minus_T1", "T4", "T1"),
)
PAIR_KEY_FIELDS = (
    "event_id",
    "upstream_model_family",
    "model_family",
    "representation_seed",
    "decision_seed",
    "investor_group",
)
EVENT_MODEL_GROUP_DECISION_FIELDS = (
    "event_id",
    "model_family",
    "investor_group",
    "decision_seed",
)
CELL_KEY_FIELDS = (
    "event_id",
    "treatment_family",
    "upstream_model_family",
    "model_family",
    "representation_seed",
    "decision_seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-rows", default=str(DEFAULT_DECISION_ROWS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--text-distance-method",
        choices=("token_cosine_proxy", "sentence_transformer"),
        default="sentence_transformer",
    )
    parser.add_argument("--embedding-model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--device",
        default=None,
        help="Optional sentence-transformers device, e.g. cuda, mps, or cpu.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for smoke tests.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Batch size for sentence-transformer embedding preload.",
    )
    parser.add_argument(
        "--treatment-filter",
        default=None,
        help="Optional comma-separated treatment_family filter, e.g. B0.",
    )
    return parser.parse_args()


def parse_list_field(value: Any) -> list[str]:
    """Parse a JSON-list CSV field, falling back to a single stripped string."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [text]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if parsed is None:
        return []
    parsed_text = str(parsed).strip()
    return [parsed_text] if parsed_text else []


def join_items(items: Iterable[str]) -> str:
    return " ".join(item.strip() for item in items if item and item.strip())


def join_items_multiline(items: Iterable[str]) -> str:
    return "\n".join(item.strip() for item in items if item and item.strip())


def build_reason_text(row: dict[str, Any]) -> str:
    return join_items_multiline(parse_list_field(row.get("key_reasons")))


def build_uncertainty_text(row: dict[str, Any]) -> str:
    return join_items_multiline(parse_list_field(row.get("uncertainty_notes")))


def build_composite_reasoning_text(row: dict[str, Any]) -> str:
    reason_text = build_reason_text(row)
    uncertainty_text = build_uncertainty_text(row)
    if reason_text and uncertainty_text:
        return f"Reasons:\n{reason_text}\n\nUncertainty:\n{uncertainty_text}"
    if reason_text:
        return f"Reasons:\n{reason_text}"
    if uncertainty_text:
        return f"Uncertainty:\n{uncertainty_text}"
    return ""


def build_reasoning_texts(row: dict[str, Any]) -> dict[str, str]:
    return {
        "expressed_reason_text": build_reason_text(row),
        "uncertainty_frame_text": build_uncertainty_text(row),
        "composite_reasoning_text": build_composite_reasoning_text(row),
    }


def normalize_text(text: str) -> str:
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(cleaned.split())


def lexical_feature_counter(text: str) -> Counter[str]:
    normalized = normalize_text(text)
    counter: Counter[str] = Counter()
    for token in normalized.split():
        counter[f"w:{token}"] += 1
    compact = normalized.replace(" ", "_")
    if compact:
        if len(compact) < 3:
            counter[f"c:{compact}"] += 1
        else:
            for idx in range(len(compact) - 2):
                counter[f"c:{compact[idx:idx + 3]}"] += 1
    return counter


def cosine_distance_from_counters(left: Counter[str], right: Counter[str]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    dot = sum(left[key] * right.get(key, 0.0) for key in left)
    cosine = max(0.0, min(1.0, dot / (left_norm * right_norm)))
    return 1.0 - cosine


def normalized_cosine_distance(left: Any, right: Any) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        left_float = float(left_value)
        right_float = float(right_value)
        dot += left_float * right_float
        left_norm += left_float * left_float
        right_norm += right_float * right_float
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    cosine = dot / math.sqrt(left_norm * right_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return (1.0 - cosine) / 2.0


def cosine_distance(left: Any, right: Any) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        left_float = float(left_value)
        right_float = float(right_value)
        dot += left_float * right_float
        left_norm += left_float * left_float
        right_norm += right_float * right_float
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    cosine = dot / math.sqrt(left_norm * right_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return 1.0 - cosine


def pairwise_cosine_distances(vectors: list[Any]) -> list[float]:
    return [cosine_distance(left, right) for left, right in combinations(vectors, 2)]


def mean_pairwise_distance(items: list[Any], distance_fn: Callable[[Any, Any], float]) -> float | None:
    if len(items) < 2:
        return None
    return mean(distance_fn(left, right) for left, right in combinations(items, 2))


def safe_mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None and not math.isnan(value)]
    return None if not clean else mean(clean)


def parse_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


def load_decision_rows(path: Path, max_rows: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row.update(build_reasoning_texts(row))
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def filter_decision_rows(
    rows: list[dict[str, Any]],
    treatment_filter: str | None,
) -> list[dict[str, Any]]:
    if not treatment_filter:
        return rows
    allowed = {item.strip() for item in treatment_filter.split(",") if item.strip()}
    if not allowed:
        return rows
    return [row for row in rows if str(row.get("treatment_family", "")) in allowed]


class TextDistanceEngine:
    def __init__(self, method_name: str, model_name: str | None = None, model: Any | None = None) -> None:
        self.method_name = method_name
        self.model_name = model_name
        self._model = model
        self._embedding_cache: dict[str, Any] = {}

    @classmethod
    def build(
        cls,
        method: str,
        embedding_model: str | None,
        device: str | None = None,
    ) -> "TextDistanceEngine":
        if method == "token_cosine_proxy":
            return cls("token_char_ngram_cosine_distance_proxy")
        if not embedding_model:
            raise ValueError("--embedding-model is required for sentence_transformer")
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(embedding_model, device=device)
        return cls("sentence_transformer_normalized_cosine_distance", embedding_model, model)

    def mean_pairwise_text_distance(self, texts: list[str]) -> float | None:
        if len(texts) < 2:
            return None
        if self._model is None:
            counters = [lexical_feature_counter(text) for text in texts]
            return mean_pairwise_distance(counters, cosine_distance_from_counters)
        return mean_pairwise_distance(
            [self.embedding_for_text(text) for text in texts],
            normalized_cosine_distance,
        )

    def embedding_for_text(self, text: str) -> Any:
        if not text.strip():
            return []
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self._model.encode(text, convert_to_numpy=True)
        return self._embedding_cache[text]

    def preload_texts(self, texts: Iterable[str], batch_size: int = 64) -> int:
        if self._model is None:
            return 0
        unique_texts = sorted({text for text in texts if text.strip() and text not in self._embedding_cache})
        if not unique_texts:
            return 0
        embeddings = self._model.encode(
            unique_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        for text, embedding in zip(unique_texts, embeddings, strict=True):
            self._embedding_cache[text] = embedding
        return len(unique_texts)


def semantic_distance_for_texts(texts: list[str], engine: TextDistanceEngine) -> float | None:
    if len(texts) < 2:
        return None
    if engine._model is None:
        return engine.mean_pairwise_text_distance(texts)

    distances: list[float] = []
    for left, right in combinations(texts, 2):
        left_empty = not left.strip()
        right_empty = not right.strip()
        if left_empty and right_empty:
            distances.append(0.0)
        elif left_empty or right_empty:
            distances.append(1.0)
        else:
            distances.append(
                normalized_cosine_distance(
                    engine.embedding_for_text(left),
                    engine.embedding_for_text(right),
                )
            )
    return mean(distances)


def investor_group_rows(rows: list[dict[str, Any]], investor_group: str) -> list[dict[str, Any]]:
    if investor_group == "all_profiles":
        return [row for row in rows if row.get("profile_group") in {"retail", "institutional"}]
    return [row for row in rows if row.get("profile_group") == investor_group]


def compute_cell_metric(
    rows: list[dict[str, Any]],
    investor_group: str,
    engine: TextDistanceEngine,
) -> dict[str, Any]:
    base = rows[0]
    subset = investor_group_rows(rows, investor_group)
    profile_count = len(subset)
    pair_count = profile_count * (profile_count - 1) // 2

    result: dict[str, Any] = {
        "event_id": base.get("event_id", ""),
        "ticker": base.get("ticker", ""),
        "company": base.get("company", ""),
        "market_cap_group": base.get("market_cap_group", ""),
        "treatment_family": base.get("treatment_family", ""),
        "upstream_model_family": base.get("upstream_model_family", ""),
        "model_family": base.get("model_family", ""),
        "representation_seed": base.get("representation_seed", ""),
        "decision_seed": base.get("decision_seed", ""),
        "investor_group": investor_group,
        "profile_count": profile_count,
        "pair_count": pair_count,
        "text_distance_method": engine.method_name,
        "embedding_model": engine.model_name or "",
    }
    for channel in TEXT_CHANNELS:
        field_name = f"{channel}_text"
        texts = [row[field_name] for row in subset]
        nonempty_count = sum(1 for text in texts if text.strip())
        result[f"{channel}_nonempty_share"] = (
            None if profile_count == 0 else nonempty_count / profile_count
        )
        result[f"{channel}_mean_chars"] = safe_mean([float(len(text)) for text in texts])
        result[f"{channel}_semantic_distance"] = semantic_distance_for_texts(texts, engine)
    return result


def build_cell_metrics(rows: list[dict[str, Any]], engine: TextDistanceEngine) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in CELL_KEY_FIELDS)
        grouped[key].append(row)

    cell_rows: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        for investor_group in INVESTOR_GROUPS:
            subset = investor_group_rows(group_rows, investor_group)
            if len(subset) >= 2:
                cell_rows.append(compute_cell_metric(group_rows, investor_group, engine))
    return cell_rows


def build_treatment_means(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cell_rows:
        grouped[(row["treatment_family"], row["investor_group"])].append(row)

    output: list[dict[str, Any]] = []
    for (treatment, investor_group), rows in sorted(grouped.items()):
        out: dict[str, Any] = {
            "treatment_family": treatment,
            "investor_group": investor_group,
            "n_cells": len(rows),
            "n_events": len({row["event_id"] for row in rows}),
        }
        for channel in TEXT_CHANNELS:
            out[f"{channel}_semantic_distance"] = safe_mean(
                [parse_float(row[f"{channel}_semantic_distance"]) for row in rows]
            )
            out[f"{channel}_nonempty_share"] = safe_mean(
                [parse_float(row[f"{channel}_nonempty_share"]) for row in rows]
            )
            out[f"{channel}_mean_chars"] = safe_mean(
                [parse_float(row[f"{channel}_mean_chars"]) for row in rows]
            )
        output.append(out)
    return output


def unique_row_or_none(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[0] if len(rows) == 1 else None


def add_contrast_row(
    output: list[dict[str, Any]],
    *,
    key_payload: dict[str, Any],
    contrast_name: str,
    comparison_row: dict[str, Any],
    baseline_row: dict[str, Any],
) -> None:
    out: dict[str, Any] = {
        **key_payload,
        "contrast": contrast_name,
        "comparison": comparison_row["treatment_family"],
        "baseline": baseline_row["treatment_family"],
    }
    for channel in TEXT_CHANNELS:
        comparison_value = parse_float(comparison_row[f"{channel}_semantic_distance"])
        baseline_value = parse_float(baseline_row[f"{channel}_semantic_distance"])
        out[f"{channel}_comparison"] = comparison_value
        out[f"{channel}_baseline"] = baseline_value
        out[f"{channel}_delta"] = (
            None
            if comparison_value is None or baseline_value is None
            else comparison_value - baseline_value
        )
    output.append(out)


def build_contrasts(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    by_event_model_group_decision: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in cell_rows:
        key = tuple(str(row.get(field, "")) for field in PAIR_KEY_FIELDS)
        by_key[key][row["treatment_family"]] = row
        shared_key = tuple(str(row.get(field, "")) for field in EVENT_MODEL_GROUP_DECISION_FIELDS)
        by_event_model_group_decision[shared_key].append(row)

    output: list[dict[str, Any]] = []

    # Same-upstream contrasts preserve the historical T2/T3/T4 pairing.
    for key, treatment_rows in by_key.items():
        key_payload = dict(zip(PAIR_KEY_FIELDS, key, strict=True))
        for contrast_name, comparison, baseline in (
            ("T3_minus_T2", "T3", "T2"),
            ("T4_minus_T2", "T4", "T2"),
            ("T4_minus_T3", "T4", "T3"),
        ):
            if comparison not in treatment_rows or baseline not in treatment_rows:
                continue
            add_contrast_row(
                output,
                key_payload=key_payload,
                contrast_name=contrast_name,
                comparison_row=treatment_rows[comparison],
                baseline_row=treatment_rows[baseline],
            )

    # B0/T1 do not share the upstream_model_family values used by T2/T3/T4.
    # Mirror compute_diversity_metrics.py: pair them by event, receiver model,
    # investor group, and downstream decision seed, repeating B0/T1 across each
    # comparison representation path where appropriate.
    for key, rows in sorted(by_event_model_group_decision.items()):
        base_payload = dict(zip(EVENT_MODEL_GROUP_DECISION_FIELDS, key, strict=True))
        t1_row = unique_row_or_none([row for row in rows if row["treatment_family"] == "T1"])
        b0_row = unique_row_or_none([row for row in rows if row["treatment_family"] == "B0"])

        if t1_row is not None and b0_row is not None:
            add_contrast_row(
                output,
                key_payload={
                    **base_payload,
                    "upstream_model_family": b0_row["upstream_model_family"],
                    "representation_seed": b0_row["representation_seed"],
                },
                contrast_name="B0_minus_T1",
                comparison_row=b0_row,
                baseline_row=t1_row,
            )

        if b0_row is None:
            continue
        for comparison_family in ("T2", "T3", "T4"):
            comparison_rows = [
                row for row in rows if row["treatment_family"] == comparison_family
            ]
            for comparison_row in sorted(
                comparison_rows,
                key=lambda item: (
                    str(item["upstream_model_family"]),
                    str(item["representation_seed"]),
                ),
            ):
                add_contrast_row(
                    output,
                    key_payload={
                        **base_payload,
                        "upstream_model_family": comparison_row["upstream_model_family"],
                        "representation_seed": comparison_row["representation_seed"],
                    },
                    contrast_name=f"{comparison_family}_minus_B0",
                    comparison_row=comparison_row,
                    baseline_row=b0_row,
                )
    return output


def summarize_contrasts(contrast_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in contrast_rows:
        grouped[(row["contrast"], row["investor_group"])].append(row)

    output: list[dict[str, Any]] = []
    for (contrast, investor_group), rows in sorted(grouped.items()):
        out: dict[str, Any] = {
            "contrast": contrast,
            "investor_group": investor_group,
            "n_pairs": len(rows),
            "n_events": len({row["event_id"] for row in rows}),
        }
        for channel in TEXT_CHANNELS:
            deltas = [parse_float(row[f"{channel}_delta"]) for row in rows]
            clean_deltas = [value for value in deltas if value is not None]
            out[f"{channel}_mean_delta"] = safe_mean(clean_deltas)
            out[f"{channel}_positive_share"] = (
                None
                if not clean_deltas
                else sum(1 for value in clean_deltas if value > 0.0) / len(clean_deltas)
            )
        output.append(out)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    decision_rows_path = Path(args.decision_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = TextDistanceEngine.build(
        args.text_distance_method,
        args.embedding_model,
        args.device,
    )
    decision_rows = filter_decision_rows(
        load_decision_rows(decision_rows_path, args.max_rows),
        args.treatment_filter,
    )
    preloaded_embeddings = engine.preload_texts(
        (
            row[field_name]
            for row in decision_rows
            for field_name in (
                "expressed_reason_text",
                "uncertainty_frame_text",
                "composite_reasoning_text",
            )
        ),
        batch_size=args.embedding_batch_size,
    )
    cell_rows = build_cell_metrics(decision_rows, engine)
    treatment_means = build_treatment_means(cell_rows)
    contrast_rows = build_contrasts(cell_rows)
    contrast_summary = summarize_contrasts(contrast_rows)

    write_csv(output_dir / "reasoning_cell_metrics.csv", cell_rows)
    write_csv(output_dir / "reasoning_treatment_means.csv", treatment_means)
    write_csv(output_dir / "reasoning_contrasts.csv", contrast_rows)
    write_csv(output_dir / "reasoning_contrast_summary.csv", contrast_summary)
    write_summary(
        output_dir / "run_summary.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "decision_rows": str(decision_rows_path),
            "output_dir": str(output_dir),
            "total_decision_rows_loaded": len(decision_rows),
            "cell_metric_rows_written": len(cell_rows),
            "contrast_rows_written": len(contrast_rows),
            "text_distance_method": engine.method_name,
            "embedding_model": engine.model_name,
            "device": args.device,
            "max_rows": args.max_rows,
            "treatment_filter": args.treatment_filter,
            "embedding_batch_size": args.embedding_batch_size,
            "preloaded_embeddings": preloaded_embeddings,
            "formulas": {
                "expressed_reason_text": "join(key_reasons)",
                "uncertainty_frame_text": "join(uncertainty_notes)",
                "composite_reasoning_text": "join(key_reasons + uncertainty_notes)",
                "semantic_distance": "mean_{i<j} (1 - cos(e_i, e_j)) / 2 for non-empty text pairs; empty-empty=0; empty-nonempty=1",
                "lexical_proxy_distance": "mean_{i<j} 1 - cosine(token_and_character_3gram_count_i, token_and_character_3gram_count_j)",
                "contrast_delta": "comparison semantic_distance - baseline semantic_distance",
            },
        },
    )

    print(f"loaded_rows={len(decision_rows)}")
    print(f"preloaded_embeddings={preloaded_embeddings}")
    print(f"cell_rows={len(cell_rows)}")
    print(f"contrast_rows={len(contrast_rows)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
