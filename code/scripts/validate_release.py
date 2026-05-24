#!/usr/bin/env python3
"""Validate the public artifact release surface."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


MANIFEST_NAME = "result_table_manifest.csv"
REQUIRED_RELEASE_FILES = (
    "README.md",
    "code/README.md",
    "code/code_manifest.csv",
    "code/schemas/release_schema.json",
    "code/scripts/treatments.py",
    "code/scripts/metrics.py",
    "code/scripts/build_result_manifest.py",
    "code/scripts/validate_release.py",
    f"results/{MANIFEST_NAME}",
)
FORBIDDEN_HEADER_MARKERS = (
    "reviewer",
    "annotator",
    "audit_path",
    "payload_path",
    "local_path",
    "internal_path",
    "source_path",
    "provider_output_path",
)


def forbidden_text_markers() -> tuple[str, ...]:
    return (
        "/" + "Users/",
        "/" + "private" + "/" + "tmp",
        "Eric" + "0801",
        "chiuyi" + "ting",
        "pilot" + "_",
        "preflight" + "_2026",
        "2026 " + "pilot",
        "pre-" + "main-run",
    )


def default_release_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=default_release_root())
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows_and_header(path: Path) -> tuple[list[str], int]:
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.reader(input_file)
        header = next(reader, [])
        return header, sum(1 for _ in reader)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def text_files(root: Path) -> list[Path]:
    suffixes = {".csv", ".json", ".jsonl", ".md", ".py", ".txt"}
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in suffixes
    ]


def validate_required_files(root: Path, problems: list[str]) -> None:
    for relative in REQUIRED_RELEASE_FILES:
        path = root / relative
        if not path.exists():
            problems.append(f"missing required release file: {relative}")


def validate_manifest(root: Path, problems: list[str]) -> None:
    results_root = root / "results"
    manifest_path = results_root / MANIFEST_NAME
    if not manifest_path.exists():
        problems.append(f"missing manifest: results/{MANIFEST_NAME}")
        return

    rows = read_manifest(manifest_path)
    manifest_paths = {row.get("release_path", "") for row in rows}
    actual_paths = {
        path.relative_to(results_root).as_posix()
        for path in results_root.rglob("*.csv")
        if path.name != MANIFEST_NAME
    }
    for release_path in sorted(actual_paths - manifest_paths):
        problems.append(f"manifest omits released CSV: {release_path}")
    for release_path in sorted(manifest_paths - actual_paths):
        problems.append(f"manifest lists missing CSV: {release_path}")

    for row in rows:
        release_path = row.get("release_path", "")
        csv_path = results_root / release_path
        if not csv_path.exists():
            continue
        header, row_count = csv_rows_and_header(csv_path)
        if not header:
            problems.append(f"{release_path}: empty CSV header")
        if row_count == 0:
            problems.append(f"{release_path}: empty released CSV")
        expected_rows = row.get("row_count_excluding_header", "")
        if expected_rows != str(row_count):
            problems.append(
                f"{release_path}: row-count mismatch "
                f"(manifest={expected_rows}, actual={row_count})"
            )
        expected_sha = row.get("sha256", "")
        actual_sha = sha256_file(csv_path)
        if expected_sha != actual_sha:
            problems.append(
                f"{release_path}: sha256 mismatch "
                f"(manifest={expected_sha}, actual={actual_sha})"
            )
        lower_header = [field.lower() for field in header]
        for field in lower_header:
            for marker in FORBIDDEN_HEADER_MARKERS:
                if marker in field:
                    problems.append(f"{release_path}: public CSV contains internal column {field}")


def validate_code_manifest(root: Path, problems: list[str]) -> None:
    code_root = root / "code"
    manifest_path = code_root / "code_manifest.csv"
    if not manifest_path.exists():
        problems.append("missing code/code_manifest.csv")
        return
    with manifest_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    listed = {row.get("path", "") for row in rows if row.get("path")}
    actual = {
        path.relative_to(code_root).as_posix()
        for path in code_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name != "code_manifest.csv"
    }
    for relative in sorted(listed):
        if not (code_root / relative).exists():
            problems.append(f"code manifest lists missing file: {relative}")
    for relative in sorted(actual - listed):
        problems.append(f"code manifest omits public code file: {relative}")


def validate_text_markers(root: Path, problems: list[str]) -> None:
    for path in text_files(root):
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in forbidden_text_markers():
            if marker in text:
                problems.append(f"{relative}: forbidden public marker {marker!r}")


def main() -> int:
    args = parse_args()
    release_root = args.release_root.resolve()
    problems: list[str] = []
    if not release_root.exists():
        raise FileNotFoundError(release_root)
    validate_required_files(release_root, problems)
    validate_manifest(release_root, problems)
    validate_code_manifest(release_root, problems)
    validate_text_markers(release_root, problems)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    print(f"release OK: {release_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
