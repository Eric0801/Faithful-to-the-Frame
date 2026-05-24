# Code Reference

This directory contains a cleaned reference implementation for the public
artifact release. It is intentionally smaller than the operational workspace:
batch submission scripts, provider repair scripts, local caches, and one-off
analysis notebooks are excluded.

The code here documents the algorithmic surfaces needed to interpret and check
the paper artifacts:

- `scripts/treatments.py`: deterministic treatment rendering and prompt-job
  construction for T1, T2, T3, T4, and B0.
- `scripts/metrics.py`: evidence-to-action metric computation from validated
  downstream decision rows.
- `scripts/build_result_manifest.py`: recomputes released-table row counts and
  SHA-256 checksums from the public `results/` directory.
- `scripts/validate_release.py`: validates the public release surface,
  including manifest consistency, non-empty CSVs, and absence of internal path
  or reviewer columns.
- `schemas/release_schema.json`: compact field contracts for release
  manifests, metric inputs, source packets, evidence banks, and hidden-outcome
  joins.

The released CSV tables under `../results/` remain the canonical paper
artifacts. These scripts are the compact reference path for how the treatment,
metric, result-manifest, and release-validation layers are defined.

## Release Validation

From the release root:

```bash
python3 code/scripts/validate_release.py --release-root .
python3 code/scripts/build_result_manifest.py --results-root results --check
```

`validate_release.py` checks only the public artifact surface. It does not
require private provider logs, batch-status directories, repair scripts, local
caches, or hidden operational paths.

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
