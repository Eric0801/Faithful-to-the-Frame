# Code

This directory contains the cleaned public execution path for the paper
artifact. The scripts are lightly organized copies of the project pipeline, not
a toy reimplementation. Provider batch submission, retry/status folders, local
caches, and one-off operational scripts are excluded.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r code/requirements.txt
```

Mock mode uses only the Python standard library. For provider-backed reruns,
set an API key and edit the model fields in `code/config.example.json` or
`code/config.full_rerun.example.json`:

```bash
cp code/env.example .env
export OPENAI_API_KEY=replace_with_your_api_key
```

The config exposes `api_base_url`, `api_key_env`, `upstream_model`,
`upstream_model_family`, `downstream_model_families`, and
`downstream_model_override`, so users can swap OpenAI-compatible providers
or model names without editing code.

## One-Command Run

From the release root:

```bash
python3 code/scripts/run_experiment.py \
  --config code/config.example.json \
  --work-dir runs/main_94_mock \
  --mode mock \
  --overwrite
```

`config.example.json` points to the released 94-event model-facing inputs under
`data/model_facing/main_94/` and caps the mock run for quick inspection.

For an uncapped provider-backed rerun, start from:

```bash
export OPENAI_API_KEY=replace_with_your_api_key
python3 code/scripts/run_experiment.py \
  --config code/config.full_rerun.example.json \
  --work-dir runs/provider_full \
  --mode openai-compatible \
  --overwrite
```

Provider-backed reruns can be expensive. Adjust profiles, treatments, seeds,
model families, and `limits` in the config before launching.

## Pipeline Scripts

- `run_experiment.py`: orchestrates the public pipeline.
- `render_treatments.py`: renders T1/T4 deterministic views and T2/T3 upstream
  prompt jobs from source packets and canonical evidence banks.
- `run_representation_harness.py`: executes T2/T3 upstream prompt jobs in mock
  or OpenAI-compatible mode.
- `build_downstream_requests.py`: builds role-conditioned downstream receiver
  prompts from rendered treatment rows.
- `run_downstream_harness.py`: executes downstream decision prompts in mock or
  OpenAI-compatible mode.
- `validate_decision_outputs.py`: validates downstream decision JSONL rows
  against the expected schema.
- `compute_diversity_metrics.py`: computes E2A event-level metrics, treatment
  contrasts, and optional quality readouts with evaluation-only CAR labels.
- `canonicalize_decision_outputs.py`: deterministic alias normalization utility
  for provider output fields when needed.

## Inputs And Outputs

Model-facing inputs:

```text
data/model_facing/main_94/source_packets/
data/model_facing/main_94/canonical_evidence_banks/
data/model_facing/main_94/sample_manifest.csv
```

Evaluation-only inputs:

```text
data/evaluation_only/car_1_5_outcomes.csv
```

Hidden outcomes are joined only during metric computation. They are not used by
treatment rendering, upstream representation generation, or downstream prompt
construction.
