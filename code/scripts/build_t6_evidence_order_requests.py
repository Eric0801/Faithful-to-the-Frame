#!/usr/bin/env python3
"""Build four-model downstream requests for frozen T6 order randomization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_downstream_requests import build_prompt_parts, cache_key_for_row, parse_csv_list, profile_map
from experiment_run_profiles import ALL_PROFILE_IDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "t6_evidence_order_randomization_20260711"
)
DEFAULT_PAYLOAD_DIR = DEFAULT_ROOT / "t6_payloads_event_level"
DEFAULT_MODELS = ("gpt-5.2", "claude-sonnet-4.5", "qwen3-235b-a22b", "deepseek-v3.1")
TREATMENT_NAME = "T6_canonical_evidence_order_randomized"
UPSTREAM_MODEL_FAMILY = "deterministic_evidence_order_randomization"
EXPECTED_EVENTS = 94


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-dir", default=str(DEFAULT_PAYLOAD_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_ROOT))
    parser.add_argument("--model-families", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--profiles", default=",".join(ALL_PROFILE_IDS))
    parser.add_argument("--decision-seed", type=int, default=1)
    parser.add_argument("--representation-seed", type=int, default=0)
    return parser.parse_args()


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_payloads(payload_dir: Path, representation_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(payload_dir.glob("evt_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("treatment") != TREATMENT_NAME:
            continue
        if int(payload.get("representation_seed", -1)) != representation_seed:
            continue
        if payload.get("upstream_model_family") != UPSTREAM_MODEL_FAMILY:
            raise ValueError(f"unexpected upstream family in {path}")
        if not str(payload.get("rendered_text", "")).strip():
            raise ValueError(f"missing rendered_text in {path}")
        payload["_source_path"] = project_relative(path)
        rows.append(payload)
    if len(rows) != EXPECTED_EVENTS:
        raise ValueError(f"expected {EXPECTED_EVENTS} T6 payloads; found {len(rows)}")
    if len({str(row["event_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate T6 event payload")
    return rows


def main() -> int:
    args = parse_args()
    payloads = load_payloads(Path(args.payload_dir), args.representation_seed)
    output_dir = Path(args.output_dir)
    profiles = profile_map(parse_csv_list(args.profiles))
    models = parse_csv_list(args.model_families)
    if not models or not profiles:
        raise ValueError("at least one model and profile are required")
    jobs: list[dict[str, Any]] = []
    for payload in payloads:
        for model_family in models:
            for profile_id, profile in profiles.items():
                cache_prefix, cache_suffix, prompt = build_prompt_parts(
                    str(payload["rendered_text"]), profile, args.decision_seed
                )
                job = {
                    "event_id": str(payload["event_id"]),
                    "treatment": TREATMENT_NAME,
                    "upstream_model_family": UPSTREAM_MODEL_FAMILY,
                    "profile": profile_id,
                    "profile_group": str(profile["profile_group"]),
                    "model_family": model_family,
                    "representation_seed": args.representation_seed,
                    "decision_seed": args.decision_seed,
                    "prompt": prompt,
                    "cache_prefix": cache_prefix,
                    "cache_suffix": cache_suffix,
                    "treatment_payload_path": payload["_source_path"],
                    "permutation_version": payload["permutation_version"],
                    "manifest_event_seed_sha256": payload["manifest_event_seed_sha256"],
                }
                job["prompt_cache_key"] = cache_key_for_row(job, profile_id)
                jobs.append(job)
    expected_jobs = EXPECTED_EVENTS * len(models) * len(profiles)
    if len(jobs) != expected_jobs:
        raise ValueError(f"expected {expected_jobs} jobs; built {len(jobs)}")
    write_jsonl(output_dir / "downstream_requests_t6_all.jsonl", jobs)
    for model in models:
        safe = model.replace(".", "_").replace("-", "_")
        write_jsonl(
            output_dir / f"downstream_requests_t6_{safe}.jsonl",
            [job for job in jobs if job["model_family"] == model],
        )
    summary = {
        "treatment": TREATMENT_NAME,
        "upstream_model_family": UPSTREAM_MODEL_FAMILY,
        "events": EXPECTED_EVENTS,
        "model_families": models,
        "profile_count": len(profiles),
        "decision_seed": args.decision_seed,
        "representation_seed": args.representation_seed,
        "total_jobs": len(jobs),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "t6_request_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
