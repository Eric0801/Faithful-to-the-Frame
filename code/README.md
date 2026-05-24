# Code Reference

This directory contains a cleaned reference implementation for the public
artifact release. It is intentionally smaller than the operational workspace:
batch submission scripts, provider repair scripts, local caches, and one-off
analysis notebooks are excluded.

The code here documents the algorithmic surfaces needed to interpret the paper
artifacts:

- `scripts/run_experiment.py`: end-to-end runner from source/evidence inputs to
  treatments, upstream summaries, downstream decisions, and metrics.
- `scripts/treatments.py`: deterministic treatment rendering and prompt-job
  construction for T1, T2, T3, T4, and B0.
- `scripts/downstream_decisions.py`: downstream receiver prompt construction
  and deterministic normalization of model decision JSON into decision rows.
- `scripts/metrics.py`: evidence-to-action metric computation from validated
  downstream decision rows.
- `schemas/experiment_schema.json`: compact field contracts for experiment
  manifests, source packets, evidence banks, downstream requests, downstream
  decision rows, and hidden-outcome joins.

The released CSV tables under `../results/` remain the canonical paper
artifacts. These scripts are the compact reference path for how the treatment,
downstream decision, and metric layers are defined.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r code/requirements.txt
```

For API-backed execution, set an API key and edit `code/config.example.json`:

```bash
cp code/env.example .env
export OPENAI_API_KEY=replace_with_your_api_key
```

The config exposes `api_base_url`, `api_key_env`, `model`,
`upstream_model_family`, and `downstream_model_families`, so reviewers can swap
providers or model names without editing code.

## Run The Pipeline

From the release root:

```bash
python3 code/scripts/run_experiment.py \
  --config code/config.example.json \
  --work-dir runs/demo \
  --mode mock \
  --overwrite
```

To use an OpenAI-compatible endpoint instead of mock mode:

```bash
export OPENAI_API_KEY=replace_with_your_api_key
python3 code/scripts/run_experiment.py \
  --config code/config.example.json \
  --work-dir runs/provider_demo \
  --mode openai_compatible \
  --overwrite
```

The example inputs are intentionally tiny; they exercise the full code path
without claiming to reproduce the released paper estimates. To run a larger
replication, edit `inputs.source_packets_dir`, `inputs.evidence_banks_dir`,
model settings, seeds, and profile/model axes in `code/config.example.json`.

## Treatment Reference

`scripts/treatments.py` expects source packets and canonical evidence banks as
JSON files keyed by `event_id`. It renders:

- `T1_raw_public_information`: direct source-packet text view.
- `T2_shared_summary`: one source-grounded upstream prompt job per
  event/representation seed.
- `T3_independent_summary`: one source-grounded upstream prompt job per
  event/profile/representation seed.
- `T4_full_structured_evidence_ledger`: deterministic full ledger over every
  canonical evidence unit.
- `B0_canonical_evidence_only`: deterministic canonical-bank serialization sent
  directly downstream.

Example:

```bash
python3 code/scripts/treatments.py render \
  --source-packets-dir path/to/source_packets \
  --evidence-banks-dir path/to/evidence_banks \
  --output-dir path/to/treatments \
  --treatments T1,T2,T3,T4,B0
```

Smoke-test example:

```bash
python3 code/scripts/treatments.py render \
  --source-packets-dir code/examples/source_packets \
  --evidence-banks-dir code/examples/evidence_banks \
  --output-dir /tmp/e2a_treatments_smoke \
  --treatments T1,T2,T3,T4,B0 \
  --overwrite
```

## Metric Reference

`scripts/metrics.py` expects a validated downstream `decision_rows.csv`. It
computes event-level evidence-to-action readouts and treatment contrasts:

- belief dispersion: mean pairwise absolute difference in expected returns
- action diversity: action entropy and one-minus-HHI
- rationale diversity: deterministic lexical cosine distance
- source and evidence-category diversity: one-minus pairwise overlap
- optional quality readouts when hidden `CAR_1_5` values are supplied

Example:

```bash
python3 code/scripts/metrics.py \
  path/to/decision_rows.csv \
  --hidden-outcomes-csv path/to/hidden_outcomes.csv \
  --output-dir path/to/metric_outputs
```

Smoke-test example:

```bash
python3 code/scripts/metrics.py \
  code/examples/decision_rows.csv \
  --hidden-outcomes-csv code/examples/evaluation_outcomes_demo.csv \
  --output-dir /tmp/e2a_metrics_smoke
```

Hidden outcomes are joined only inside metric computation. They are not used by
the treatment renderer or prompt-job construction.
