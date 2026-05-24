#!/usr/bin/env python3
"""Rebuild or check the released result-table manifest.

The manifest is intentionally scoped to files already present under
``release/results``. It does not know about the private operational workspace,
provider batches, repair attempts, or local cache paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


MANIFEST_FIELDS = [
    "release_group",
    "release_path",
    "role",
    "artifact_use",
    "paper_mapping",
    "row_count_excluding_header",
    "sha256",
]


def default_results_root() -> Path:
    return Path(__file__).resolve().parents[2] / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=default_results_root())
    parser.add_argument("--manifest-name", default="result_table_manifest.csv")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Overwrite the manifest with recomputed row counts and SHA-256 values.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the existing manifest is not in sync with released CSV files.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.reader(input_file)
        next(reader, None)
        return sum(1 for _ in reader)


def read_existing_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as input_file:
        return {
            row["release_path"]: row
            for row in csv.DictReader(input_file)
            if row.get("release_path")
        }


def released_csv_paths(results_root: Path, manifest_name: str) -> list[Path]:
    return [
        path
        for path in sorted(results_root.rglob("*.csv"))
        if path.name != manifest_name
    ]


def build_rows(results_root: Path, manifest_name: str) -> list[dict[str, str]]:
    manifest_path = results_root / manifest_name
    existing = read_existing_manifest(manifest_path)
    rows: list[dict[str, str]] = []
    for csv_path in released_csv_paths(results_root, manifest_name):
        release_path = csv_path.relative_to(results_root).as_posix()
        previous = existing.get(release_path, {})
        rows.append(
            {
                "release_group": csv_path.parent.relative_to(results_root).as_posix(),
                "release_path": release_path,
                "role": previous.get("role", "unclassified_release_table"),
                "artifact_use": previous.get("artifact_use", ""),
                "paper_mapping": previous.get("paper_mapping", ""),
                "row_count_excluding_header": str(count_csv_rows(csv_path)),
                "sha256": sha256_file(csv_path),
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def diff_rows(
    existing: dict[str, dict[str, str]],
    recomputed: list[dict[str, str]],
) -> list[str]:
    problems: list[str] = []
    recomputed_by_path = {row["release_path"]: row for row in recomputed}
    for release_path in sorted(set(existing) - set(recomputed_by_path)):
        problems.append(f"manifest lists missing file: {release_path}")
    for release_path in sorted(set(recomputed_by_path) - set(existing)):
        problems.append(f"manifest omits released CSV: {release_path}")
    for release_path, row in sorted(recomputed_by_path.items()):
        previous = existing.get(release_path)
        if not previous:
            continue
        for field in ("row_count_excluding_header", "sha256"):
            if previous.get(field, "") != row[field]:
                problems.append(
                    f"{release_path}: {field} mismatch "
                    f"(manifest={previous.get(field, '')}, actual={row[field]})"
                )
    return problems


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    manifest_path = results_root / args.manifest_name
    if not results_root.exists():
        raise FileNotFoundError(results_root)

    rows = build_rows(results_root, args.manifest_name)
    if args.write:
        write_manifest(manifest_path, rows)

    if args.check or not args.write:
        existing = read_existing_manifest(manifest_path)
        problems = diff_rows(existing, rows)
        if problems:
            for problem in problems:
                print(f"ERROR: {problem}", file=sys.stderr)
            return 1
        print(f"manifest OK: {len(rows)} released CSV files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
