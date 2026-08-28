#!/usr/bin/env python3
"""Build T5 linguistic-deframing downstream request files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_downstream_requests import (
    build_prompt_parts,
    cache_key_for_row,
    parse_csv_list,
    profile_map,
)
from build_b0_canonical_baseline_requests import (
    TREATMENT_NAME as B0_TREATMENT_NAME,
    UPSTREAM_MODEL_FAMILY as B0_UPSTREAM_MODEL_FAMILY,
    load_b0_payloads,
    make_job as make_b0_job,
)
from experiment_run_profiles import ALL_PROFILE_IDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_T5_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "t5_linguistic_deframing_20260710"
)
DEFAULT_T5_PAYLOAD_DIR = DEFAULT_T5_DIR / "t5_payloads_event_level"
DEFAULT_B0_PAYLOAD_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "b0_canonical_baseline_20260510"
    / "b0_payloads_event_level"
)
DEFAULT_OUTPUT_DIR = DEFAULT_T5_DIR
DEFAULT_MODEL_FAMILIES = ("gpt-5.2", "claude-sonnet-4.5")
TREATMENT_NAME = "T5_linguistic_deframing"
UPSTREAM_MODEL_FAMILY = "deterministic_linguistic_deframing"
EXPECTED_MAIN_EVENTS = 94


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-payload-dir", default=str(DEFAULT_T5_PAYLOAD_DIR))
    parser.add_argument(
        "--b0-payload-dir",
        default=str(DEFAULT_B0_PAYLOAD_DIR),
        help="Canonical B0 payloads for the contemporaneous paired replay.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--model-families",
        default=",".join(DEFAULT_MODEL_FAMILIES),
        help="Comma-separated downstream model family IDs.",
    )
    parser.add_argument(
        "--profiles",
        default=",".join(ALL_PROFILE_IDS),
        help="Comma-separated investor profiles for T5.",
    )
    parser.add_argument("--decision-seed", type=int, default=1)
    parser.add_argument("--representation-seed", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_t5_payloads(
    t5_payload_dir: Path,
    representation_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(t5_payload_dir.glob("evt_*.json")):
        payload = read_json(path)
        if payload.get("treatment") != TREATMENT_NAME:
            continue
        if int(payload.get("representation_seed", -1)) != representation_seed:
            continue
        upstream_model_family = str(payload.get("upstream_model_family", "")).strip()
        if upstream_model_family != UPSTREAM_MODEL_FAMILY:
            raise ValueError(
                f"{path} has upstream_model_family={upstream_model_family!r}; "
                f"expected {UPSTREAM_MODEL_FAMILY!r}"
            )
        if not str(payload.get("rendered_text", "")).strip():
            raise ValueError(f"{path} has no rendered_text")
        payload["_source_path"] = project_relative(path)
        rows.append(payload)

    if not rows:
        raise ValueError(f"no {TREATMENT_NAME} payloads found in {t5_payload_dir}")

    event_ids = [str(row["event_id"]) for row in rows]
    duplicate_events = sorted(
        event_id for event_id in set(event_ids) if event_ids.count(event_id) > 1
    )
    if duplicate_events:
        raise ValueError(f"duplicate T5 event payloads: {', '.join(duplicate_events)}")
    return rows


def make_job(
    *,
    payload: dict[str, Any],
    model_family: str,
    profile_id: str,
    profile_group: str,
    decision_seed: int,
    cache_prefix: str,
    cache_suffix: str,
    prompt: str,
) -> dict[str, Any]:
    event_id = str(payload["event_id"])
    representation_seed = int(payload["representation_seed"])
    job = {
        "event_id": event_id,
        "treatment": TREATMENT_NAME,
        "upstream_model_family": UPSTREAM_MODEL_FAMILY,
        "profile": profile_id,
        "profile_group": profile_group,
        "model_family": model_family,
        "representation_seed": representation_seed,
        "decision_seed": decision_seed,
        "prompt": prompt,
        "cache_prefix": cache_prefix,
        "cache_suffix": cache_suffix,
        "treatment_payload_path": payload["_source_path"],
    }
    job["prompt_cache_key"] = cache_key_for_row(job, profile_id)
    return job


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def write_grouped_outputs(
    output_dir: Path,
    *,
    t5_jobs: list[dict[str, Any]],
    b0_jobs: list[dict[str, Any]],
) -> None:
    jobs = b0_jobs + t5_jobs
    write_jsonl(output_dir / "downstream_requests_t5_all.jsonl", t5_jobs)
    write_jsonl(output_dir / "downstream_requests_b0_replay_all.jsonl", b0_jobs)
    write_jsonl(output_dir / "downstream_requests_t5_b0_replay_all.jsonl", jobs)
    for model_family in sorted({str(row["model_family"]) for row in jobs}):
        safe_model_family = model_family.replace(".", "_").replace("-", "_")
        model_t5_jobs = [
            job for job in t5_jobs if job["model_family"] == model_family
        ]
        model_b0_jobs = [
            job for job in b0_jobs if job["model_family"] == model_family
        ]
        write_jsonl(
            output_dir / f"downstream_requests_t5_{safe_model_family}.jsonl",
            model_t5_jobs,
        )
        write_jsonl(
            output_dir / f"downstream_requests_b0_replay_{safe_model_family}.jsonl",
            model_b0_jobs,
        )
        write_jsonl(
            output_dir / f"downstream_requests_t5_b0_replay_{safe_model_family}.jsonl",
            model_b0_jobs + model_t5_jobs,
        )


def main() -> int:
    args = parse_args()
    t5_payload_dir = Path(args.t5_payload_dir)
    b0_payload_dir = Path(args.b0_payload_dir)
    output_dir = Path(args.output_dir)
    model_families = parse_csv_list(args.model_families)
    selected_profiles = parse_csv_list(args.profiles)
    if not model_families:
        raise ValueError("--model-families must select at least one model family")
    profiles = profile_map(selected_profiles)
    payloads = load_t5_payloads(t5_payload_dir, args.representation_seed)
    b0_payloads = load_b0_payloads(b0_payload_dir, args.representation_seed)

    t5_event_ids = {str(payload["event_id"]) for payload in payloads}
    b0_event_ids = {str(payload["event_id"]) for payload in b0_payloads}
    if t5_event_ids != b0_event_ids:
        missing_b0 = sorted(t5_event_ids - b0_event_ids)
        missing_t5 = sorted(b0_event_ids - t5_event_ids)
        raise ValueError(
            "B0/T5 event coverage mismatch: "
            f"missing B0={missing_b0}, missing T5={missing_t5}"
        )

    t5_jobs: list[dict[str, Any]] = []
    for payload in payloads:
        rendered_text = str(payload["rendered_text"])
        for model_family in model_families:
            for profile_id, profile in profiles.items():
                cache_prefix, cache_suffix, prompt = build_prompt_parts(
                    rendered_text,
                    profile,
                    args.decision_seed,
                )
                t5_jobs.append(
                    make_job(
                        payload=payload,
                        model_family=model_family,
                        profile_id=profile_id,
                        profile_group=str(profile["profile_group"]),
                        decision_seed=args.decision_seed,
                        cache_prefix=cache_prefix,
                        cache_suffix=cache_suffix,
                        prompt=prompt,
                    )
                )

    b0_jobs: list[dict[str, Any]] = []
    for payload in b0_payloads:
        rendered_text = str(payload["rendered_text"])
        for model_family in model_families:
            for profile_id, profile in profiles.items():
                cache_prefix, cache_suffix, prompt = build_prompt_parts(
                    rendered_text,
                    profile,
                    args.decision_seed,
                )
                b0_jobs.append(
                    make_b0_job(
                        payload=payload,
                        model_family=model_family,
                        profile_id=profile_id,
                        profile_group=str(profile["profile_group"]),
                        decision_seed=args.decision_seed,
                        cache_prefix=cache_prefix,
                        cache_suffix=cache_suffix,
                        prompt=prompt,
                    )
                )

    write_grouped_outputs(output_dir, t5_jobs=t5_jobs, b0_jobs=b0_jobs)
    expected_main_jobs = (
        EXPECTED_MAIN_EVENTS * len(DEFAULT_MODEL_FAMILIES) * len(ALL_PROFILE_IDS)
    )
    summary = {
        "events": len(payloads),
        "model_families": model_families,
        "profile_count": len(profiles),
        "decision_seed": args.decision_seed,
        "representation_seed": args.representation_seed,
        "b0_replay_jobs": len(b0_jobs),
        "t5_jobs": len(t5_jobs),
        "total_jobs": len(b0_jobs) + len(t5_jobs),
        "treatment": TREATMENT_NAME,
        "upstream_model_family": UPSTREAM_MODEL_FAMILY,
        "expected_main_events": EXPECTED_MAIN_EVENTS,
        "expected_main_default_jobs_per_treatment": expected_main_jobs,
        "expected_main_default_jobs_paired": expected_main_jobs * 2,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "t5_request_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
