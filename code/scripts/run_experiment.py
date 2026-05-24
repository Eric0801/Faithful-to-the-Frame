#!/usr/bin/env python3
"""Run the public E2A pipeline from released inputs.

This wrapper calls the original pipeline scripts included in this artifact:

1. render treatments from source packets and canonical evidence banks
2. execute upstream T2/T3 representation jobs
3. build downstream receiver requests
4. execute downstream decision jobs
5. validate decision outputs
6. compute E2A metrics

Use ``--mode mock`` for a credential-free end-to-end run. Use
``--mode openai-compatible`` with an API key for provider-backed reruns.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("code/config.example.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("runs/main_94_mock"))
    parser.add_argument(
        "--mode",
        choices=("mock", "openai-compatible"),
        default=None,
        help="Override config execution_mode.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def release_root() -> Path:
    return Path(__file__).resolve().parents[2]


def script(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run(args: list[str], cwd: Path) -> None:
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def optional_limit(args: list[str], flag: str, value: Any) -> list[str]:
    if value is None or value == 0:
        return args
    return args + [flag, str(value)]


def main() -> int:
    args = parse_args()
    root = release_root()
    config_path = resolve(root, args.config.as_posix())
    config = load_config(config_path)
    mode = args.mode or config.get("execution_mode", "mock")
    work_dir = resolve(root, args.work_dir.as_posix())
    if work_dir.exists() and args.overwrite:
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    inputs = config["inputs"]
    experiment = config["experiment"]
    provider = config["provider"]
    limits = config.get("limits", {})

    treatments_dir = work_dir / "01_treatments"
    upstream_outputs = treatments_dir / "rendered_upstream_outputs.jsonl"
    downstream_requests = work_dir / "02_downstream_requests.jsonl"
    downstream_outputs = work_dir / "03_downstream_outputs.jsonl"
    validation_report = work_dir / "03_downstream_outputs.validation_report.json"
    metrics_dir = work_dir / "04_metrics"

    run(
        [
            sys.executable,
            str(script("render_treatments.py")),
            "--source-packets-dir",
            str(resolve(root, inputs["source_packets_dir"])),
            "--evidence-banks-dir",
            str(resolve(root, inputs["evidence_banks_dir"])),
            "--output-dir",
            str(treatments_dir),
            "--treatments",
            ",".join(experiment["treatments"]),
            "--profiles",
            ",".join(experiment["profiles"]),
            "--representation-seeds",
            ",".join(str(seed) for seed in experiment["representation_seeds"]),
        ],
        cwd=root,
    )

    representation_command = [
        sys.executable,
        str(script("run_representation_harness.py")),
        str(treatments_dir),
        "--output-path",
        str(upstream_outputs),
        "--mode",
        mode,
        "--provider",
        provider.get("provider_label", mode),
        "--model",
        provider["upstream_model"],
        "--upstream-model-family",
        provider["upstream_model_family"],
        "--workers",
        str(provider.get("workers", 1)),
        "--api-base-url",
        provider["api_base_url"],
        "--api-key-env-var",
        provider["api_key_env"],
        "--api-temperature",
        str(provider.get("upstream_temperature", 0.2)),
        "--api-max-tokens",
        str(provider.get("upstream_max_tokens", 900)),
        "--no-resume",
    ]
    representation_command = optional_limit(
        representation_command,
        "--max-jobs",
        limits.get("max_upstream_jobs"),
    )
    run(representation_command, cwd=root)

    downstream_command = [
        sys.executable,
        str(script("build_downstream_requests.py")),
        "--rendered-treatments-dir",
        str(treatments_dir),
        "--output-path",
        str(downstream_requests),
        "--model-families",
        ",".join(provider["downstream_model_families"]),
        "--profiles",
        ",".join(experiment["profiles"]),
        "--decision-seeds",
        ",".join(str(seed) for seed in experiment["decision_seeds"]),
        "--upstream-model-families",
        provider["upstream_model_family"],
    ]
    run(downstream_command, cwd=root)

    downstream_exec = [
        sys.executable,
        str(script("run_downstream_harness.py")),
        str(downstream_requests),
        "--output-path",
        str(downstream_outputs),
        "--provider",
        mode,
        "--workers",
        str(provider.get("workers", 1)),
        "--api-base-url",
        provider["api_base_url"],
        "--api-key-env-var",
        provider["api_key_env"],
        "--api-model-override",
        provider.get("downstream_model_override", ""),
        "--api-temperature",
        str(provider.get("downstream_temperature", 0.0)),
        "--api-max-tokens",
        str(provider.get("downstream_max_tokens", 700)),
        "--overwrite",
    ]
    downstream_exec = optional_limit(
        downstream_exec,
        "--max-requests",
        limits.get("max_downstream_requests"),
    )
    run(downstream_exec, cwd=root)

    run(
        [
            sys.executable,
            str(script("validate_decision_outputs.py")),
            str(downstream_outputs),
            "--report",
            str(validation_report),
            "--source-packets",
            str(resolve(root, inputs["source_packets_dir"])),
        ],
        cwd=root,
    )

    metric_command = [
        sys.executable,
        str(script("compute_diversity_metrics.py")),
        str(downstream_outputs),
        "--output-dir",
        str(metrics_dir),
        "--event-metadata-csv",
        str(resolve(root, inputs["sample_manifest_csv"])),
    ]
    hidden_outcomes = inputs.get("hidden_outcomes_csv")
    if hidden_outcomes:
        metric_command.extend(["--hidden-outcomes-csv", str(resolve(root, hidden_outcomes))])
    run(metric_command, cwd=root)

    print(f"pipeline complete: {work_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
