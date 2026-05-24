#!/usr/bin/env python3
"""Build downstream decision request bundles from rendered treatments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiment_run_profiles import (
    INVESTOR_PROFILES,
    RUN_PROFILES,
    get_run_profile,
    is_t1_treatment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "downstream_requests.jsonl"

PROFILES = list(INVESTOR_PROFILES)
DEFAULT_MODEL_FAMILIES = ["claude-sonnet-4.5"]
DEFAULT_DECISION_SEEDS = [1, 2]
DOWNSTREAM_STATIC_INSTRUCTIONS = (
    "You are a downstream decision agent in a controlled experiment.\n\n"
    "Task:\n"
    "- Read the supplied treatment packet.\n"
    "- Then apply the supplied investor profile and decision seed.\n"
    "- Estimate expected_return_5d as a decimal return in [-0.20, 0.20].\n"
    "- Provide confidence in [0, 1].\n"
    "- Choose one action from buy, hold, sell.\n"
    "- Provide action_strength in [-1, 1].\n"
    "- Provide 2 to 5 concise key_reasons.\n"
    "- Cite evidence_used as source IDs from the packet when available.\n"
    "- Provide optional uncertainty_notes.\n"
    "- Use only information present in the treatment packet, including company, ticker, and date if supplied.\n"
    "- Do not use external knowledge, browsing, post-event outcomes, realized market reactions, hidden labels, future filings/news, or unsupported facts.\n\n"
    "Return JSON with keys:\n"
    "event_id, treatment, upstream_model_family, profile, profile_group, "
    "model_family, representation_seed, decision_seed, expected_return_5d, confidence, "
    "action, action_strength, key_reasons, evidence_used, uncertainty_notes.\n\n"
)


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seed_list(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rendered-treatments-dir",
        required=True,
        help="Directory containing rendered treatment JSON files and/or rendered JSONL outputs.",
    )
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--run-profile",
        default="",
        choices=sorted(RUN_PROFILES),
        help="Optional run profile that supplies default treatments/model axes/seeds.",
    )
    parser.add_argument(
        "--model-families",
        "--downstream-model-families",
        dest="model_families",
        default="",
        help="Comma-separated downstream model family IDs",
    )
    parser.add_argument(
        "--upstream-model-families",
        default="",
        help="Comma-separated upstream model families to retain for T2/T3 rows.",
    )
    parser.add_argument(
        "--default-upstream-model-family",
        default="",
        help=(
            "Fallback upstream model family for legacy T2/T3 representation rows "
            "that lack upstream_model_family."
        ),
    )
    parser.add_argument(
        "--decision-seeds",
        default="",
        help="Comma-separated decision seeds",
    )
    parser.add_argument(
        "--profiles",
        default="",
        help="Comma-separated subset of profile IDs",
    )
    parser.add_argument(
        "--treatments",
        default="",
        help=(
            "Optional comma-separated treatment IDs to include from rendered inputs. "
            "Use this when old and new T4 renderings coexist in the same tree."
        ),
    )
    return parser.parse_args()


def resolve_configured_defaults(args: argparse.Namespace) -> dict[str, list[Any]]:
    if args.run_profile:
        run_profile = get_run_profile(args.run_profile)
        treatments = list(run_profile.treatments)
        upstream_model_families = list(run_profile.upstream_model_families)
        model_families = list(run_profile.downstream_model_families)
        profiles = list(run_profile.profiles)
        decision_seeds = list(run_profile.decision_seeds)
    else:
        treatments = []
        upstream_model_families = []
        model_families = DEFAULT_MODEL_FAMILIES
        profiles = [item["profile"] for item in PROFILES]
        decision_seeds = DEFAULT_DECISION_SEEDS

    if args.treatments:
        treatments = parse_csv_list(args.treatments)
    if args.upstream_model_families:
        upstream_model_families = parse_csv_list(args.upstream_model_families)
    if args.model_families:
        model_families = parse_csv_list(args.model_families)
    if args.profiles:
        profiles = parse_csv_list(args.profiles)
    if args.decision_seeds:
        decision_seeds = parse_seed_list(args.decision_seeds)

    return {
        "treatments": treatments,
        "upstream_model_families": upstream_model_families,
        "model_families": model_families,
        "profiles": profiles,
        "decision_seeds": decision_seeds,
    }


def load_rendered_treatments(path: Path) -> list[dict[str, Any]]:
    rows = []
    if path.is_file():
        rows = load_rendered_treatments_from_file(path)
        if not rows:
            raise ValueError(f"no rendered treatment JSON rows found in {path}")
        return rows

    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in {".json", ".jsonl"}:
            continue
        rows.extend(load_rendered_treatments_from_file(candidate))
    if not rows:
        raise ValueError(f"no rendered treatment JSON files found in {path}")
    return rows


def load_rendered_treatments_from_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line_number, raw_line in enumerate(fh, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if not isinstance(record, dict):
                    continue
                if "rendered_text" not in record:
                    continue
                record["_source_path"] = str(path)
                record["_source_line_number"] = line_number
                rows.append(record)
        return rows

    record = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(record, dict) and "rendered_text" in record:
        record["_source_path"] = str(path)
        record["_source_line_number"] = None
        rows.append(record)
    return rows


def rendered_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    upstream_model_family = str(row.get("upstream_model_family", "")).strip()
    job_key = row.get("job_key")
    if isinstance(job_key, str) and job_key.strip():
        return ("job_key", upstream_model_family, job_key.strip())
    return (
        "cell",
        upstream_model_family,
        str(row.get("event_id", "")).strip(),
        str(row.get("treatment", "")).strip(),
        int(row.get("representation_seed", 0)),
        str(row.get("profile_id", "")).strip(),
    )


def dedupe_rendered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        latest_by_key[rendered_row_key(row)] = row
    return list(latest_by_key.values())


def filter_rendered_rows_by_treatment(
    rows: list[dict[str, Any]],
    selected_treatments: set[str],
) -> list[dict[str, Any]]:
    if not selected_treatments:
        return rows
    return [row for row in rows if str(row.get("treatment", "")).strip() in selected_treatments]


def resolve_upstream_model_family(
    row: dict[str, Any],
    expected_upstream_model_families: list[str],
    default_upstream_model_family: str,
) -> str:
    treatment = str(row.get("treatment", "")).strip()
    if is_t1_treatment(treatment):
        return "none"

    upstream_model_family = str(row.get("upstream_model_family", "")).strip()
    if upstream_model_family:
        return upstream_model_family

    if default_upstream_model_family:
        return default_upstream_model_family
    if len(expected_upstream_model_families) == 1:
        return expected_upstream_model_families[0]
    if len(expected_upstream_model_families) > 1:
        source = row.get("_source_path", "<unknown>")
        line = row.get("_source_line_number")
        location = f"{source}:{line}" if line is not None else str(source)
        raise ValueError(
            "T2/T3 rendered row lacks upstream_model_family while multiple "
            f"upstream families are expected: {location}"
        )
    return "legacy_unspecified"


def annotate_and_filter_upstream_model_family(
    rows: list[dict[str, Any]],
    expected_upstream_model_families: list[str],
    default_upstream_model_family: str,
) -> list[dict[str, Any]]:
    selected = set(expected_upstream_model_families)
    annotated: list[dict[str, Any]] = []
    for row in rows:
        upstream_model_family = resolve_upstream_model_family(
            row,
            expected_upstream_model_families,
            default_upstream_model_family,
        )
        if (
            selected
            and not is_t1_treatment(str(row.get("treatment", "")).strip())
            and upstream_model_family not in selected
        ):
            continue
        annotated_row = dict(row)
        annotated_row["upstream_model_family"] = upstream_model_family
        annotated.append(annotated_row)
    return dedupe_rendered_rows(annotated)


def profile_map(selected_ids: list[str]) -> dict[str, dict[str, Any]]:
    profiles = {item["profile"]: item for item in PROFILES}
    missing = [item for item in selected_ids if item not in profiles]
    if missing:
        raise ValueError(f"unknown profiles: {', '.join(missing)}")
    return {item: profiles[item] for item in selected_ids}


def build_cache_prefix(rendered_text: str) -> str:
    return (
        DOWNSTREAM_STATIC_INSTRUCTIONS
        + "Treatment packet:\n"
        + f"{rendered_text}\n"
    )


def build_cache_suffix(profile: dict[str, Any], decision_seed: int) -> str:
    focus = ", ".join(profile["focus"])
    return (
        "\nInvestor profile and seed:\n"
        f"Profile: {profile['profile']}\n"
        f"Profile group: {profile['profile_group']}\n"
        f"Horizon: {profile['horizon']}\n"
        f"Focus: {focus}\n"
        f"Style: {profile['style']}\n"
        f"Decision seed: {decision_seed}\n"
    )


def build_prompt_parts(
    rendered_text: str,
    profile: dict[str, Any],
    decision_seed: int,
) -> tuple[str, str, str]:
    cache_prefix = build_cache_prefix(rendered_text)
    cache_suffix = build_cache_suffix(profile, decision_seed)
    return (
        cache_prefix,
        cache_suffix,
        cache_prefix + cache_suffix,
    )


def cache_key_for_row(row: dict[str, Any], locked_profile_id: str) -> str:
    profile_component = locked_profile_id or "shared"
    return ":".join(
        [
            "downstream",
            str(row.get("upstream_model_family", "")).strip(),
            str(row.get("event_id", "")).strip(),
            str(row.get("treatment", "")).strip(),
            str(int(row.get("representation_seed", 0))),
            profile_component,
        ]
    )


def main() -> int:
    args = parse_args()
    configured = resolve_configured_defaults(args)
    rendered_dir = Path(args.rendered_treatments_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rendered_rows = filter_rendered_rows_by_treatment(
        load_rendered_treatments(rendered_dir),
        set(str(item) for item in configured["treatments"]),
    )
    rendered_rows = annotate_and_filter_upstream_model_family(
        rendered_rows,
        [str(item) for item in configured["upstream_model_families"]],
        args.default_upstream_model_family.strip(),
    )
    if not rendered_rows:
        raise ValueError("no rendered rows remain after treatment/upstream filtering")
    profiles = profile_map([str(item) for item in configured["profiles"]])
    model_families = [str(item) for item in configured["model_families"]]
    decision_seeds = [int(item) for item in configured["decision_seeds"]]

    total = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rendered_rows:
            representation_seed = int(row.get("representation_seed", 0))
            locked_profile_id = str(row.get("profile_id", "")).strip()
            if locked_profile_id:
                if locked_profile_id not in profiles:
                    raise ValueError(
                        f"rendered row references unknown profile_id: {locked_profile_id}"
                    )
                row_profiles = {locked_profile_id: profiles[locked_profile_id]}
            else:
                row_profiles = profiles
            for model_family in model_families:
                for profile_id, profile in row_profiles.items():
                    for decision_seed in decision_seeds:
                        cache_prefix, cache_suffix, prompt = build_prompt_parts(
                            row["rendered_text"],
                            profile,
                            decision_seed,
                        )
                        job = {
                            "event_id": row["event_id"],
                            "treatment": row["treatment"],
                            "upstream_model_family": row["upstream_model_family"],
                            "profile": profile_id,
                            "profile_group": profile["profile_group"],
                            "model_family": model_family,
                            "representation_seed": representation_seed,
                            "decision_seed": decision_seed,
                            "prompt": prompt,
                            "cache_prefix": cache_prefix,
                            "cache_suffix": cache_suffix,
                            "prompt_cache_key": cache_key_for_row(
                                row,
                                locked_profile_id,
                            ),
                            "treatment_payload_path": row["_source_path"],
                        }
                        if row.get("_source_line_number") is not None:
                            job["treatment_payload_line_number"] = row[
                                "_source_line_number"
                            ]
                        if locked_profile_id:
                            job["representation_profile"] = locked_profile_id
                        fh.write(json.dumps(job, ensure_ascii=True, sort_keys=True))
                        fh.write("\n")
                        total += 1

    print(f"wrote {total} downstream requests to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
