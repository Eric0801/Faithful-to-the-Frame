#!/usr/bin/env python3
"""Run the public E2A reference pipeline from config.

The runner supports two execution modes:

- ``mock``: deterministic local responses, useful for checking the full code
  path without provider credentials.
- ``openai_compatible``: calls a chat-completions endpoint configured in
  ``config.example.json`` and ``env.example``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("code/config.example.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("runs/demo"))
    parser.add_argument("--mode", choices=("mock", "openai_compatible"), default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def release_root() -> Path:
    return Path(__file__).resolve().parents[2]


def script_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for raw_line in input_file:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_command(args: list[str], cwd: Path) -> None:
    print("+ " + " ".join(args))
    subprocess.run(args, cwd=cwd, check=True)


def stable_float(token: str, low: float, high: float) -> float:
    raw = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:12], 16)
    ratio = raw / float(16**12 - 1)
    return low + (high - low) * ratio


def mock_upstream_summary(job: dict[str, Any]) -> str:
    prompt = str(job.get("prompt", ""))
    source_ids = sorted(set(part for part in prompt.replace(",", " ").split() if part.startswith(("S", "X"))))
    cited = ", ".join(source_ids[:6]) or "the supplied source IDs"
    return (
        "This source-grounded summary describes the disclosed event using only "
        f"the canonical evidence bank. It cites {cited}, separates operating "
        "claims from uncertainty, and does not provide an action recommendation."
    )


def mock_downstream_decision(request: dict[str, Any]) -> dict[str, Any]:
    key = str(request["request_key"])
    expected = stable_float(key + "::return", -0.04, 0.04)
    confidence = stable_float(key + "::confidence", 0.45, 0.78)
    action = "buy" if expected > 0.012 else "sell" if expected < -0.012 else "hold"
    strength = {"buy": 0.55, "hold": 0.0, "sell": -0.55}[action]
    prompt = str(request.get("prompt", ""))
    sources = sorted(set(part.strip("[](),.;:") for part in prompt.split() if part.startswith(("S", "X"))))
    evidence = sources[:4] or ["S001"]
    return {
        **{field: request[field] for field in request if field != "prompt"},
        "expected_return_5d": round(expected, 6),
        "confidence": round(confidence, 4),
        "action": action,
        "action_strength": strength,
        "key_reasons": [
            "The decision is based only on the supplied treatment packet.",
            "The mock runner preserves the downstream output schema for testing.",
        ],
        "evidence_used": evidence,
        "evidence_categories": ["revenue", "costs_or_expenses"],
        "uncertainty_notes": ["mock execution; replace mode with openai_compatible for provider calls"],
        "rationale": "Mock downstream decision generated deterministically from the request key.",
    }


def chat_completion(config: dict[str, Any], prompt: str) -> str:
    provider = config["provider"]
    api_base_url = provider["api_base_url"].rstrip("/")
    endpoint = api_base_url if api_base_url.endswith("/chat/completions") else api_base_url + "/chat/completions"
    api_key = os.environ.get(provider.get("api_key_env", "OPENAI_API_KEY"), "")
    if not api_key:
        raise RuntimeError(f"missing API key env var: {provider.get('api_key_env', 'OPENAI_API_KEY')}")
    payload = {
        "model": provider["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": provider.get("temperature", 0.0),
        "max_tokens": provider.get("max_tokens", 700),
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    retries = int(provider.get("max_retries", 2))
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=float(provider.get("timeout_seconds", 60))) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"provider call failed after {attempt + 1} attempts: {exc}") from exc
            time.sleep(float(provider.get("retry_backoff_seconds", 2.0)) * (attempt + 1))
    raise AssertionError("unreachable")


def execute_upstream(config: dict[str, Any], treatments_dir: Path, mode: str) -> Path:
    output_path = treatments_dir / "rendered_upstream_outputs.jsonl"
    rows: list[dict[str, Any]] = []
    limit = config.get("limits", {}).get("max_upstream_jobs")
    for prompt_path in sorted(treatments_dir.glob("prompt_jobs_*.jsonl")):
        for job in read_jsonl(prompt_path):
            if limit is not None and len(rows) >= int(limit):
                break
            rendered_text = (
                mock_upstream_summary(job)
                if mode == "mock"
                else chat_completion(config, str(job["prompt"]))
            )
            rows.append(
                {
                    "event_id": job["event_id"],
                    "treatment": job["treatment"],
                    "profile_id": job.get("profile_id", ""),
                    "representation_seed": job.get("representation_seed", 0),
                    "upstream_model_family": config["provider"].get("upstream_model_family", config["provider"]["model"]),
                    "rendered_text": rendered_text,
                }
            )
        if limit is not None and len(rows) >= int(limit):
            break
    write_jsonl(output_path, rows)
    print(f"wrote {len(rows)} upstream rendered rows to {output_path}")
    return output_path


def execute_downstream(config: dict[str, Any], requests_path: Path, output_path: Path, mode: str) -> None:
    requests = read_jsonl(requests_path)
    limit = config.get("limits", {}).get("max_downstream_jobs")
    if limit is not None:
        requests = requests[: int(limit)]
    rows: list[dict[str, Any]] = []
    for request in requests:
        if mode == "mock":
            rows.append(mock_downstream_decision(request))
        else:
            content = chat_completion(config, str(request["prompt"]))
            rows.append({**{field: request[field] for field in request if field != "prompt"}, "content": content})
    write_jsonl(output_path, rows)
    print(f"wrote {len(rows)} downstream provider-output rows to {output_path}")


def main() -> int:
    args = parse_args()
    root = release_root()
    config_path = resolve_path(root, args.config.as_posix()) if not args.config.is_absolute() else args.config
    config = load_json(config_path)
    mode = args.mode or config.get("mode", "mock")
    work_dir = args.work_dir if args.work_dir.is_absolute() else root / args.work_dir
    if work_dir.exists() and args.overwrite:
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    source_packets = resolve_path(root, config["inputs"]["source_packets_dir"])
    evidence_banks = resolve_path(root, config["inputs"]["evidence_banks_dir"])
    hidden_outcomes = resolve_path(root, config["inputs"].get("hidden_outcomes_csv"))
    treatments_dir = work_dir / "01_treatments"
    requests_path = work_dir / "02_downstream_requests.jsonl"
    provider_outputs = work_dir / "03_provider_outputs.jsonl"
    decision_rows = work_dir / "04_decision_rows.csv"
    metrics_dir = work_dir / "05_metrics"

    run_command(
        [
            sys.executable,
            str(script_path("treatments.py")),
            "render",
            "--source-packets-dir",
            str(source_packets),
            "--evidence-banks-dir",
            str(evidence_banks),
            "--output-dir",
            str(treatments_dir),
            "--treatments",
            ",".join(config["experiment"]["treatments"]),
            "--profiles",
            ",".join(config["experiment"]["profiles"]),
            "--representation-seeds",
            ",".join(str(seed) for seed in config["experiment"]["representation_seeds"]),
            "--overwrite",
        ],
        cwd=root,
    )
    execute_upstream(config, treatments_dir, mode)
    run_command(
        [
            sys.executable,
            str(script_path("downstream_decisions.py")),
            "build-requests",
            "--rendered-treatments-dir",
            str(treatments_dir),
            "--output-path",
            str(requests_path),
            "--model-families",
            ",".join(config["provider"]["downstream_model_families"]),
            "--profiles",
            ",".join(config["experiment"]["profiles"]),
            "--decision-seeds",
            ",".join(str(seed) for seed in config["experiment"]["decision_seeds"]),
        ],
        cwd=root,
    )
    execute_downstream(config, requests_path, provider_outputs, mode)
    run_command(
        [
            sys.executable,
            str(script_path("downstream_decisions.py")),
            "normalize-outputs",
            "--provider-outputs-jsonl",
            str(provider_outputs),
            "--output-csv",
            str(decision_rows),
            "--strict",
        ],
        cwd=root,
    )
    metric_command = [
        sys.executable,
        str(script_path("metrics.py")),
        str(decision_rows),
        "--output-dir",
        str(metrics_dir),
    ]
    if hidden_outcomes is not None:
        metric_command.extend(["--hidden-outcomes-csv", str(hidden_outcomes)])
    run_command(metric_command, cwd=root)
    print(f"pipeline complete: {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
