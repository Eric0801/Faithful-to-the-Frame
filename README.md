# Faithful to the Frame

### Source-Framing Propagation in LLM-Agent Decision Workflows

This is the camera-ready reproducibility artifact for the EMNLP Main paper *Faithful to the Frame: Source-Framing Propagation in LLM-Agent Decision Workflows*. It releases the outcome-blind inputs, pipeline code, validation tools, and paper-facing result tables for the evidence-to-action (E2A) evaluation.

E2A asks how a public earnings-event disclosure is transformed by an LLM evidence interface, and how that representation changes the evidence a role-conditioned downstream LLM cites, reasons from, believes, and acts on.

## Start here

This release supports four practical entry points:

1. Inspect the released [94-event model-facing inputs](data/model_facing/main_94) and their canonical source packets.
2. Run the pipeline in a credential-free deterministic mock mode.
3. Recompute the paper-facing analyses from the released decision records and evaluation-only outcome labels.
4. Audit the source-grounding and outcome-blindness boundary before using the artifact in further work.

The public release contains the inputs and traces needed to inspect and recompute the reported analyses. A fresh run against an external model provider is intentionally **not** expected to be bitwise-identical: provider models and APIs can change over time. The released result tables and provenance artifacts are the reproducibility record for the camera-ready numerical claims.

## Quick start

The following command runs the full pipeline with deterministic mock providers; no API credentials or event outcomes are exposed to the model-facing stage.

```bash
python3 code/scripts/run_experiment.py \\
  --config code/config.example.json \\
  --work-dir runs/main_94_mock \\
  --mode mock \\
  --overwrite
```

For setup details, provider configuration, and the pipeline output layout, see [`code/README.md`](code/README.md). [`code/PAPER_CODE.md`](code/PAPER_CODE.md) maps each of the paper's analyses to its implementation script and expected inputs/outputs.

## Released data boundary

The main sample has **94 earnings events**: 47 large-cap and 47 small/mid-cap issuers, with at most one event per company. Events were accepted on or after 2026-01-01. The release deliberately separates data that a model may see from evaluation labels:

| Location | Purpose | Model-visible? |
| --- | --- | --- |
| [`data/model_facing/main_94`](data/model_facing/main_94) | Source packets, canonical evidence banks, and treatment inputs | Yes |
| [`data/evaluation_only/car_1_5_outcomes.csv`](data/evaluation_only/car_1_5_outcomes.csv) | `CAR_1_5` outcome labels used only during metric computation | No |
| [`results/calibration`](results/calibration) | Compact pre-main-run calibration outputs | Separate validation use |

Source packets contain event-time public evidence and identifiers. Pre-event market context is retained as model-facing context; post-event outcomes are kept exclusively in `evaluation_only/` and joined only after downstream decisions have been produced. The [outcome-blindness audit](code/scripts/audit_outcome_blindness.py) and the script map in [`code/PAPER_CODE.md`](code/PAPER_CODE.md) document and check this boundary.

## Experimental conditions

All main representations are grounded in the same canonical outcome-blind evidence bank. They differ in how that evidence is selected, organized, and presented to downstream decision agents.

| Condition | Representation | Role in the paper |
| --- | --- | --- |
| T1 | Raw disclosure/source packet | Source-faithful baseline |
| T2 | One shared neutral-analyst summary per event | Shared synthesized evidence interface |
| T3 | Role-conditioned independent summaries | Personalized synthesized evidence interface |
| B0 | Canonical evidence only, without synthesis | Structured no-synthesis anchor |
| T4 | Full structured evidence ledger | Targeted mechanism follow-up |
| T2* / T3* | Summary robustness variants without neutral-analyst framing | Framing-control robustness checks |
| T5 | Linguistic deframing intervention | Narrow mitigation-mechanism follow-up |
| T6 | Evidence-order randomization | Salience/order sensitivity control |
| E1 | Four-model raw-recovery validation | Independent raw-source recovery evidence |

The [treatment-generation scripts](code/scripts) and [treatment prompt templates](prompts/prompt_templates.md) make the representation choices inspectable. No treatment is intended to contain an explicit buy/hold/sell recommendation or post-event return outcome.

## Results and camera-ready controls

Paper-facing tables are preserved under [`results/`](results). The top-level [result manifest](results/result_table_manifest.csv) lists the released tables, their checksums, and their paper-claim mappings.

| Release component | Where to inspect it |
| --- | --- |
| Main E2A result tables | [`results/main`](results/main) |
| Calibration/validation outputs | [`results/calibration`](results/calibration) |
| T2*/T3* robustness results | [`results/prompt_sensitivity`](results/prompt_sensitivity) |
| T4 structured-ledger follow-up | [`results/t4_mechanism`](results/t4_mechanism) |
| Camera-ready controls | [`results/camera_ready_controls`](results/camera_ready_controls) |

The camera-ready control additions are deliberately scoped as controls and mechanism probes, rather than retroactive replacements for the main study:

- **T5 — linguistic deframing.** The [protocol](docs/t5_linguistic_deframing_followup.md) and [summary](results/camera_ready_controls/t5_followup_summary.json) test whether targeted de-framing edits alter the downstream pattern. This is a narrow intervention, not a claim of a general mitigation method.
- **T6 — evidence-order randomization.** The [protocol](docs/t6_evidence_order_randomization_followup.md) and [TOST/action-accuracy output](results/camera_ready_controls/t6_action_accuracy_tost.json) evaluate whether the reported effects depend on source ordering or salience.
- **E1 — raw-recovery validation.** The [validation record](docs/e1_four_model_recovery_validation.md) and [test](tests/test_validate_e1_prior_hierarchy_outputs.py) document the four-model recovery/provenance check without duplicating restricted raw provider traces.

For an overview of the experimental controls and their relation to the camera-ready revision, read [`docs/camera_ready_controls.md`](docs/camera_ready_controls.md).

## Reproducing and checking outputs

Run the narrow validation check for the E1 provenance record:

```bash
python3 tests/test_validate_e1_prior_hierarchy_outputs.py
```

Validate a produced decision JSONL against its schema and source packet IDs:

```bash
python3 code/scripts/validate_decision_outputs.py INPUT.jsonl \\
  --report reproduced/source_validation.json \\
  --source-packets data/model_facing/main_94/source_packets
```

Additional checks include treatment/input validation, representation audits, and outcome-blindness audits. Their commands and expected output contracts are listed in [`code/PAPER_CODE.md`](code/PAPER_CODE.md).

## Repository map

```text
code/       runnable pipeline, prompts, configuration, and script-to-paper map
data/       released model-facing inputs, evaluation-only labels, calibration data
docs/       experimental protocols, validation notes, and control documentation
results/    curated paper-facing tables and camera-ready control summaries
tests/      independent validation tests for released artifact records
```

The operational research workspace contains provider logs, intermediate caches, and other non-release artifacts. They are excluded here to keep the public artifact auditable and to preserve the model-facing/evaluation-only boundary.

## Reuse and limitations

Please cite the paper when using this artifact. Public source materials, third-party datasets, and model providers remain subject to their respective terms. This repository does not grant rights to redistribute content that is not included here; consult the source metadata in the released packets for attribution.
