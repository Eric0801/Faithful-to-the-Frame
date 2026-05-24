# Code Reference

This directory contains a cleaned reference implementation for the public
artifact release. It is intentionally smaller than the operational workspace:
batch submission scripts, provider repair scripts, local caches, and one-off
analysis notebooks are excluded.

The code here documents the two algorithmic surfaces needed to interpret the
paper artifacts:

- `scripts/treatments.py`: deterministic treatment rendering and prompt-job
  construction for T1, T2, T3, T4, and B0.
- `scripts/metrics.py`: evidence-to-action metric computation from validated
  downstream decision rows.

The released CSV tables under `../results/` remain the canonical paper
artifacts. These scripts are the compact reference path for how the treatment
and metric layers are defined.

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

Hidden outcomes are joined only inside metric computation. They are not used by
the treatment renderer or prompt-job construction.
