#!/usr/bin/env python3
"""Build B0 canonical-evidence-only downstream request files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_downstream_requests import build_prompt_parts, parse_csv_list, profile_map
from experiment_run_profiles import ALL_PROFILE_IDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_B0_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "b0_canonical_baseline_20260510"
    / "b0_payloads_event_level"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean_strict_predata_2026_main"
    / "b0_canonical_baseline_20260510"
)
DEFAULT_MODEL_FAMILIES = (
    "claude-sonnet-4.5",
    "gpt-5.2",
    "qwen3-235b-a22b",
    "deepseek-v3.1",
)
TREATMENT_NAME = "B0_canonical_evidence_only"
UPSTREAM_MODEL_FAMILY = "deterministic_canonical_evidence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-payload-dir", default=str(DEFAULT_B0_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--model-families",
        default=",".join(DEFAULT_MODEL_FAMILIES),
        help="Comma-separated downstream model family IDs.",
    )
    parser.add_argument(
        "--profiles",
        default=",".join(ALL_PROFILE_IDS),
        help="Comma-separated investor profiles for B0.",
    )
    parser.add_argument("--decision-seed", type=int, default=1)
    parser.add_argument("--representation-seed", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_b0_payloads(b0_dir: Path, representation_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(b0_dir.glob("evt_*.json")):
        payload = read_json(path)
        if payload.get("treatment") != TREATMENT_NAME:
            continue
        if int(payload.get("representation_seed", -1)) != representation_seed:
            continue
        payload["_source_path"] = project_relative(path)
        rows.append(payload)
    if not rows:
        raise ValueError(f"no {TREATMENT_NAME} payloads found in {b0_dir}")
    event_ids = [str(row["event_id"]) for row in rows]
    duplicate_events = sorted(
        event_id for event_id in set(event_ids) if event_ids.count(event_id) > 1
    )
    if duplicate_events:
        raise ValueError(f"duplicate B0 event payloads: {', '.join(duplicate_events)}")
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
    return {
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
        "prompt_cache_key": ":".join(
            [
                "downstream",
                UPSTREAM_MODEL_FAMILY,
                event_id,
                TREATMENT_NAME,
                str(representation_seed),
                f"profile:{profile_id}",
            ]
        ),
        "treatment_payload_path": payload["_source_path"],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def write_grouped_outputs(output_dir: Path, jobs: list[dict[str, Any]]) -> None:
    write_jsonl(output_dir / "downstream_requests_b0_profile.jsonl", jobs)
    write_jsonl(output_dir / "downstream_requests_b0_all.jsonl", jobs)
    for model_family in sorted({str(row["model_family"]) for row in jobs}):
        safe = model_family.replace(".", "_").replace("-", "_")
        model_rows = [row for row in jobs if row["model_family"] == model_family]
        write_jsonl(output_dir / f"downstream_requests_b0_{safe}.jsonl", model_rows)


def main() -> int:
    args = parse_args()
    b0_dir = Path(args.b0_payload_dir)
    output_dir = Path(args.output_dir)
    model_families = parse_csv_list(args.model_families)
    selected_profiles = parse_csv_list(args.profiles)
    profiles = profile_map(selected_profiles)
    payloads = load_b0_payloads(b0_dir, args.representation_seed)

    jobs: list[dict[str, Any]] = []
    for payload in payloads:
        rendered_text = str(payload["rendered_text"])
        for model_family in model_families:
            for profile_id, profile in profiles.items():
                cache_prefix, cache_suffix, prompt = build_prompt_parts(
                    rendered_text,
                    profile,
                    args.decision_seed,
                )
                jobs.append(
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

    write_grouped_outputs(output_dir, jobs)
    summary = {
        "events": len(payloads),
        "model_families": model_families,
        "profile_count": len(profiles),
        "decision_seed": args.decision_seed,
        "representation_seed": args.representation_seed,
        "profile_jobs": len(jobs),
        "total_jobs": len(jobs),
        "treatment": TREATMENT_NAME,
        "upstream_model_family": UPSTREAM_MODEL_FAMILY,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "b0_request_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
