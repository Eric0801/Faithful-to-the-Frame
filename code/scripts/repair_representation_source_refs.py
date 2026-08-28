#!/usr/bin/env python3
"""Deterministically repair source-reference typos in representation outputs.

This tool only changes citation tokens. It does not rewrite claim text. Rows
with unresolved invalid source-like tokens are copied unchanged and emitted as
targeted rerun prompt jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "clean_strict_predata_2026_main"
DEFAULT_T2_PROMPT_JOBS = (
    DEFAULT_DATA_ROOT / "treatments_entity_visible" / "prompt_jobs_T2_shared_summary.jsonl"
)
DEFAULT_T3_PROMPT_JOBS = (
    DEFAULT_DATA_ROOT / "treatments_entity_visible" / "prompt_jobs_T3_independent_summary.jsonl"
)

SOURCE_ID_PATTERN = re.compile(r"^(?:S\d{3,}|X\d{3,}|CTX\d+)$")
SOURCE_TOKEN_PATTERN = re.compile(r"\b(?:S\d+|X\d+|CTX\d+)\b")
SOURCE_LIKE_TRAILING_CHARS = set(".,;:)]}-\u2013\u2014")
RERUN_TREATMENTS = {"T2_shared_summary", "T3_independent_summary"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation-output", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--report-csv", required=True)
    parser.add_argument("--rerun-output-dir", required=True)
    parser.add_argument("--t2-prompt-jobs", default=str(DEFAULT_T2_PROMPT_JOBS))
    parser.add_argument("--t3-prompt-jobs", default=str(DEFAULT_T3_PROMPT_JOBS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {key: value for key, value in row.items() if not key.startswith("_")}
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def project_relative(path: str | Path) -> str:
    path_obj = Path(path)
    try:
        return str(path_obj.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path_obj)


def row_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row.get("event_id", "")),
        str(row.get("treatment", "")),
        int(row.get("representation_seed", 0)),
        str(row.get("profile_id", "") or ""),
    )


def prompt_job_lookup(paths: list[Path]) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    jobs: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    duplicates: list[tuple[str, str, int, str]] = []
    for path in paths:
        for row in read_jsonl(path):
            key = row_key(row)
            if key in jobs:
                duplicates.append(key)
            row["_source_path"] = str(path)
            jobs[key] = row
    if duplicates:
        sample = ", ".join(str(key) for key in duplicates[:5])
        raise ValueError(f"duplicate prompt-job keys: {sample}")
    return jobs


def evidence_bank_source_ids(path: str | Path) -> set[str]:
    bank_path = resolve_project_path(path)
    payload = json.loads(bank_path.read_text(encoding="utf-8"))
    units = payload.get("evidence_units")
    if not isinstance(units, list):
        raise ValueError(f"{bank_path} has no evidence_units list")
    return {
        str(source_id)
        for unit in units
        if isinstance(unit, dict)
        for source_id in unit.get("source_ids", [])
        if source_id
    }


def is_source_like_token_context(text: str, start: int, end: int, token: str) -> bool:
    if SOURCE_ID_PATTERN.match(token):
        return True
    if token.startswith("CTX"):
        return True
    cursor = end
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text):
        return True
    return text[cursor] in SOURCE_LIKE_TRAILING_CHARS


def repair_candidate(token: str, valid_source_ids: set[str]) -> str | None:
    if token in valid_source_ids:
        return token
    match = re.fullmatch(r"([SX])(\d+)", token)
    if not match:
        return None
    prefix, digits = match.groups()
    number = int(digits)
    if number <= 0:
        return None
    same_prefix = f"{prefix}{number:03d}"
    if same_prefix in valid_source_ids:
        return same_prefix
    other_prefix = "X" if prefix == "S" else "S"
    cross_prefix = f"{other_prefix}{number:03d}"
    if cross_prefix in valid_source_ids:
        return cross_prefix
    return None


def repair_rendered_text(
    text: str,
    valid_source_ids: set[str],
) -> tuple[str, list[dict[str, str]], list[str], list[str]]:
    repairs: list[dict[str, str]] = []
    unresolved: list[str] = []
    ignored_non_citations: list[str] = []

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if not is_source_like_token_context(text, match.start(), match.end(), token):
            ignored_non_citations.append(token)
            return token
        if token in valid_source_ids:
            return token
        candidate = repair_candidate(token, valid_source_ids)
        if candidate and candidate != token:
            repairs.append({"from": token, "to": candidate})
            return candidate
        unresolved.append(token)
        return token

    repaired = SOURCE_TOKEN_PATTERN.sub(replace, text)
    unique_unresolved = sorted(set(unresolved))
    unique_ignored = sorted(set(ignored_non_citations))
    return repaired, repairs, unique_unresolved, unique_ignored


def write_report_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "line_number",
        "event_id",
        "treatment",
        "representation_seed",
        "profile_id",
        "status",
        "repair_count",
        "repairs",
        "unresolved_source_refs",
        "ignored_non_citation_tokens",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def ensure_writable(paths: list[Path], overwrite: bool) -> None:
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")


def main() -> int:
    args = parse_args()
    representation_output = resolve_project_path(args.representation_output)
    output_path = resolve_project_path(args.output_path)
    report_json = resolve_project_path(args.report_json)
    report_csv = resolve_project_path(args.report_csv)
    rerun_output_dir = resolve_project_path(args.rerun_output_dir)
    ensure_writable([output_path, report_json, report_csv], args.overwrite)
    if rerun_output_dir.exists() and any(rerun_output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{rerun_output_dir} exists; pass --overwrite")

    prompt_jobs = prompt_job_lookup(
        [resolve_project_path(args.t2_prompt_jobs), resolve_project_path(args.t3_prompt_jobs)]
    )
    rows = read_jsonl(representation_output)
    output_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    unresolved_keys: set[tuple[str, str, int, str]] = set()
    source_id_cache: dict[str, set[str]] = {}

    for row in rows:
        bank_path = str(row.get("evidence_bank_path", ""))
        if bank_path not in source_id_cache:
            source_id_cache[bank_path] = evidence_bank_source_ids(bank_path)
        valid_source_ids = source_id_cache[bank_path]
        rendered_text = str(row.get("rendered_text", ""))
        repaired_text, repairs, unresolved, ignored = repair_rendered_text(
            rendered_text,
            valid_source_ids,
        )
        output_row = {key: value for key, value in row.items() if not key.startswith("_")}
        status = "unchanged"
        if unresolved:
            status = "needs_rerun"
            if row.get("treatment") in RERUN_TREATMENTS:
                unresolved_keys.add(row_key(row))
        elif repairs:
            status = "repaired"
            output_row["rendered_text"] = repaired_text
            metadata = output_row.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            metadata = dict(metadata)
            metadata["deterministic_source_ref_repair"] = {
                "repair_count": len(repairs),
                "repairs": repairs,
                "script": "scripts/repair_representation_source_refs.py",
            }
            output_row["metadata"] = metadata
        output_rows.append(output_row)
        if repairs or unresolved or ignored:
            report_rows.append(
                {
                    "line_number": row.get("_line_number"),
                    "event_id": row.get("event_id", ""),
                    "treatment": row.get("treatment", ""),
                    "representation_seed": row.get("representation_seed", ""),
                    "profile_id": row.get("profile_id", "") or "",
                    "status": status,
                    "repair_count": len(repairs),
                    "repairs": json.dumps(repairs, ensure_ascii=True, sort_keys=True),
                    "unresolved_source_refs": ";".join(unresolved),
                    "ignored_non_citation_tokens": ";".join(ignored),
                }
            )

    missing_jobs = sorted(unresolved_keys - set(prompt_jobs))
    if missing_jobs:
        sample = ", ".join(str(key) for key in missing_jobs[:5])
        raise ValueError(f"missing prompt jobs for unresolved rows: {sample}")

    rerun_jobs = [prompt_jobs[key] for key in sorted(unresolved_keys)]
    rerun_by_treatment = {treatment: [] for treatment in sorted(RERUN_TREATMENTS)}
    for job in rerun_jobs:
        rerun_by_treatment[str(job.get("treatment", ""))].append(job)

    write_jsonl(output_path, output_rows)
    rerun_output_dir.mkdir(parents=True, exist_ok=True)
    t2_rerun_path = rerun_output_dir / "prompt_jobs_T2_shared_summary.source_ref_rerun.jsonl"
    t3_rerun_path = rerun_output_dir / "prompt_jobs_T3_independent_summary.source_ref_rerun.jsonl"
    write_jsonl(t2_rerun_path, rerun_by_treatment["T2_shared_summary"])
    write_jsonl(t3_rerun_path, rerun_by_treatment["T3_independent_summary"])
    write_report_csv(report_csv, report_rows)

    summary = {
        "source_representation_output": project_relative(representation_output),
        "repaired_output_path": project_relative(output_path),
        "row_count": len(rows),
        "rows_with_report_entries": len(report_rows),
        "repaired_row_count": sum(1 for row in report_rows if row["status"] == "repaired"),
        "needs_rerun_row_count": sum(1 for row in report_rows if row["status"] == "needs_rerun"),
        "ignored_non_citation_row_count": sum(
            1 for row in report_rows if row["ignored_non_citation_tokens"]
        ),
        "unresolved_unique_key_count": len(unresolved_keys),
        "rerun_prompt_jobs": {
            "T2_shared_summary": {
                "path": project_relative(t2_rerun_path),
                "row_count": len(rerun_by_treatment["T2_shared_summary"]),
            },
            "T3_independent_summary": {
                "path": project_relative(t3_rerun_path),
                "row_count": len(rerun_by_treatment["T3_independent_summary"]),
            },
        },
        "repair_rules": [
            "zero-pad S/X citation tokens when the padded source ID exists in the same evidence bank",
            "map Snnn to Xnnn or Xnnn to Snnn only when the same numeric ID exists in the opposite source family and the original token is absent",
            "ignore malformed tokens in non-citation context, e.g. product names followed by ordinary words",
            "do not delete or rewrite unresolved citation tokens; emit those rows for targeted rerun",
        ],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"rows={len(rows)}")
    print(f"report_entries={len(report_rows)}")
    print(f"repaired_rows={summary['repaired_row_count']}")
    print(f"needs_rerun_rows={summary['needs_rerun_row_count']}")
    print(f"repaired_output={output_path}")
    print(f"report_json={report_json}")
    print(f"rerun_dir={rerun_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
