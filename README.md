# Faithful to the Frame: Source-Framing Propagation in LLM-Agent Decision Workflows

This repository accompanies the paper **"Faithful to the Frame:
Source-Framing Propagation in LLM-Agent Decision Workflows."**

It contains the data artifacts, treatment definitions, result tables, and
reproducibility metadata/code references used to study source-framing propagation in
LLM-agent decision workflows. The project introduces an evidence-to-action
propagation (E2A) evaluation: given public earnings-event source material, it
traces how different evidence representations affect what LLM receivers cite,
reason from, believe, and decide.

The full operational workspace contains provider runs, repair attempts, batch
status directories, local caches, and intermediate scratch files. Those are not
part of this release directory. This directory provides the clean artifact map:
where the paper's main datasets, treatment families, result tables, and
follow-up analyses are located.

## What This Repository Contains

The experiment studies how institutionally authored source framing propagates
through LLM-mediated evidence interfaces into downstream investment decisions.

The core pipeline is:

1. collect and screen public earnings-event source material
2. run the 12-event November-December 2025 calibration set for pipeline
   validation and profile-separability checks
3. construct the 94-event post-cutoff main sample
4. build source packets and canonical evidence banks under outcome blindness
5. apply leakage controls before treatment execution
6. render the five representation regimes from the same evidence substrate
7. run role-conditioned downstream LLM receivers
8. compute evidence, rationale, belief, action, and quality readouts

```mermaid
flowchart TD
    A["**Step 1. Dataset Construction**<br/>Public earnings-event sources:<br/>Form 8-K, Exhibit 99.1, XBRL companyfacts"] --> B["**Main-sample screening**<br/>94 post-cutoff events:<br/>47 large-cap / 47 small-mid"]
    A --> HC["**Step 2. Calibration**<br/>November-December 2025 calibration set:<br/>12 events, 6 large / 6 small-mid"]
    HC --> HD["**Pipeline validation**<br/>profile separability, parsing, provider, metrics"]
    B --> G["**Step 3. Main-Sample Construction and Blindness**<br/>calibration set excluded from primary estimates"]
    HD --> G
    G --> C["**Source packets**<br/>entity-visible, outcome-blind event-time evidence"]
    C --> D["**Canonical evidence banks**<br/>claims, categories, values, attribution, source IDs"]
    D --> LS["**Leakage controls**<br/>source-ID validation, bank consistency,<br/>outcome-blindness audits, model-family memory probes"]

    LS --> T1["**Step 4. T1 raw disclosure**<br/>source-elicited baseline before synthesis"]
    LS --> U["**Step 4. Upstream LLM summaries**<br/>neutral analyst framing"]
    U --> T2["**T2 shared summary**<br/>one summary per event"]
    U --> T3["**T3 independent summaries**<br/>six role-conditioned summaries per event"]
    LS --> T4["**Step 4. T4 structured evidence ledger**<br/>no-synthesis format anchor"]
    LS --> B0["**Step 5. B0 canonical evidence-only**<br/>source-faithful no-synthesis baseline"]
    LS --> US["**Step 6. Robustness variant**<br/>upstream summaries without neutral-analyst framing"]
    US --> S["**T2\* / T3\***<br/>upstream-summary robustness variant"]

    T1 --> R["**Step 7. Shared downstream stage**<br/>same role-conditioned receivers and decision schema"]
    T2 --> R
    T3 --> R
    T4 --> R
    B0 --> R
    S --> R

    R --> M["**Step 8. Metric computation**<br/>evidence, rationale, belief, action, quality"]
    H["**Hidden outcomes**<br/>CAR_1_5 evaluation only"] -. "evaluation join only" .-> M
    R --> V["**Validation and human audit**<br/>schema, source IDs, evidence and representation checks"]
    M --> O["**Curated result tables**<br/>release/results/"]
    V --> O

    O --> OM["**main/**<br/>T1/T2/T3 primary claims"]
    O --> OC["**calibration/**<br/>2025 calibration set"]
    O --> OB["**b0_followup/**<br/>canonical evidence baseline"]
    O --> OT["**t4_followup/**<br/>structured ledger main condition"]
    O --> OS["**prompt_sensitivity/**<br/>neutral-analyst ablation"]
    O --> OA["**appendix_diagnostics/**<br/>formal auxiliary checks"]
    O --> OH["**human_audit/**<br/>evidence and representation validation"]

    classDef data fill:#edf2f7,stroke:#4a5568,color:#1a202c;
    classDef cal fill:#ecfdf5,stroke:#047857,color:#1a202c;
    classDef main fill:#e6fffa,stroke:#2c7a7b,color:#1a202c;
    classDef follow fill:#fff7ed,stroke:#c05621,color:#1a202c;
    classDef robust fill:#f5f3ff,stroke:#6b46c1,color:#1a202c;
    classDef downstream fill:#e0f2fe,stroke:#0369a1,color:#1a202c;
    classDef eval fill:#fefcbf,stroke:#b7791f,color:#1a202c;
    class A,B,C,D data;
    class HC,HD,G,LS,OC cal;
    class T1,U,T2,T3,T4 main;
    class B0 follow;
    class US,S robust;
    class R downstream;
    class H,M,V,O,OM,OB,OT,OS,OA,OH eval;
```

## Dataset

The main experiment uses 94 earnings-event observations from public,
entity-visible SEC filing materials:

- 47 large-cap events
- 47 small/mid-cap events
- one event per issuer
- event window: accepted at or after 2026-01-01
- primary evaluation outcome: CAR_1_5

Entity-visible means that company identities and event dates may be visible in
the released source material. The relevant experimental separation is that
hidden outcomes are not part of the model-facing source packets, treatments,
upstream summaries, or downstream prompts.

## Representation Regimes

| Regime | Role in paper | Description |
| --- | --- | --- |
| T1 | main baseline | Raw disclosure/source packet passed directly downstream. |
| T2 | main treatment | Shared upstream summary: one summary per event shared by downstream agents. |
| T3 | main treatment | Independent upstream summaries: six role-conditioned summaries are generated per event, so each downstream role condition receives its own summary. |
| B0 | main-text follow-up | Canonical evidence-only baseline, used to separate source framing from synthesis effects. |
| T4 | main mechanism condition | Structured evidence ledger, used to test whether a more explicit evidence structure changes the bottleneck. |
| T2*/T3* | robustness | Upstream-summary variants that remove the neutral-analyst framing from T2/T3. |

## Regime And Downstream Decision Counts

All regimes enter the same downstream stage: the same role-conditioned receiver
definitions, decision schema, and metric computation are used across main,
follow-up, and robustness conditions. The table below separates representation
construction from downstream decision counts.

| Regime | Representation construction over 94 events | Downstream decision count | Count basis | Role |
| --- | --- | ---: | --- | --- |
| T1 | 94 raw source packets; no upstream LLM generation | 2,256 | 94 events x 6 roles x 4 downstream model families x 1 direct source condition seed | Main baseline |
| T2 | 94 shared neutral-analyst upstream summaries | 9,024 | 94 events x 6 roles x 4 downstream model families x 4 upstream summary model families | Main shared-summary condition |
| T3 | 564 neutral-analyst upstream summaries; 6 role-conditioned summaries per event | 9,024 | 94 events x 6 roles x 4 downstream model families x 4 upstream summary model families | Main independent-summary condition |
| T4 | 94 deterministic structured evidence ledgers; no upstream LLM generation | 9,024 | 94 events x 6 roles x 4 downstream model families x 4 ledger-rendering/model-family cells | Main mechanism condition |
| B0 | 94 deterministic canonical evidence-only renderings; no upstream LLM generation | 2,256 | 94 events x 6 roles x 4 downstream model families x 1 canonical evidence condition seed | Main-text baseline follow-up |
| T2* | 94 upstream summaries with neutral-analyst framing removed | 9,024 | same downstream design as T2 | Robustness: neutral-analyst ablation |
| T3* | 564 upstream summaries with neutral-analyst framing removed; 6 role-conditioned summaries per event | 9,024 | same downstream design as T3 | Robustness: neutral-analyst ablation |

The T1/T2/T3/T4 downstream counts are reported in
`results/t4_followup/quality_by_treatment_with_t4.csv`. B0 and T2*/T3* use the
same downstream stage and are summarized in their corresponding result-table
groups.

## Current Release Layout

```text
release/
  README.md
  code/
    README.md
    code_manifest.csv
    schemas/
      release_schema.json
    examples/
    scripts/
      build_result_manifest.py
      treatments.py
      metrics.py
      validate_release.py
  results/
    result_table_manifest.csv
    main/
    calibration/
    b0_followup/
    t4_followup/
    prompt_sensitivity/
    appendix_diagnostics/
    human_audit/
```

`results/result_table_manifest.csv` lists every released table, row count,
SHA-256 checksum, table role, and paper-claim mapping.

`code/` contains a compact public reproducibility layer for treatment
rendering, metric computation, result-manifest checking, release validation,
schema reference, and smoke-test fixtures. It intentionally excludes provider
batch submission, repair, rescue, local cache, and one-off operational scripts.

To validate the public artifact surface from the release root:

```bash
python3 code/scripts/validate_release.py --release-root .
python3 code/scripts/build_result_manifest.py --results-root results --check
```

## Results By Paper Claim

### Calibration

Location:

```text
results/calibration/
```

Key files:

- `calibration_2025_selection_summary.csv`
- `calibration_2025_selected_events.csv`
- `calibration_2025_provider_metric_summary.csv`
- `calibration_2025_profile_stochasticity_summary.csv`

The 2025 near-cutoff calibration set, originally referred to in some
internal files as a holdout set, contains 12 events from 2025-11-01 to
2025-12-31, balanced 6 large-cap / 6 small-mid-cap. Following the paper, it is
used for pipeline validation and profile-separability checks. It is not part of
the 94-event 2026+ main sample and is excluded from all primary estimates.

The full 2025 calibration working directories contain provider outputs, hidden
outcome joins, and local operational paths. The release keeps compact CSV
readouts and selected-event metadata with internal paths omitted. These are
design and QA artifacts, not confirmatory main-result estimates.

### Main Results: T1/T2/T3 Evidence-to-Action Bottleneck

Location:

```text
results/main/
```

Key tables:

- `stage2_t2_t3_primary_full.csv`: primary T2 vs T3 contrasts for belief,
  rationale, source, category, and action metrics.
- `stage2_quality_by_treatment_full.csv`: treatment-level quality guardrails.
- `stage1_summary_full.csv`: Stage 1 source/evidence compression summary.
- `stage2_by_model_cell_full.csv`: model-cell breakdown.

Use these tables for the paper's main claims about shared vs independent
summaries and whether representation changes diversity without obvious quality
collapse.

### B0 Follow-Up: Canonical Evidence-Only Baseline

Location:

```text
results/b0_followup/
```

Key tables:

- `b0_treatment_contrasts_20260510.csv`
- `b0_downstream_treatment_means_20260510.csv`
- `b0_by_receiver_model_20260510.csv`

B0 is a main-text follow-up. It is not just an appendix table. It helps
separate effects caused by source framing from effects caused by LLM synthesis.

### T4 Main Mechanism Condition: Structured Evidence Ledger

Location:

```text
results/t4_followup/
```

Key tables:

- `t4_full_treatment_means.csv`
- `t4_full_event_bootstrap_contrasts.csv`
- `quality_by_treatment_with_t4.csv`

T4 is part of the main experiment as a mechanism-oriented condition. It tests
whether a fuller structured evidence ledger changes the observed
evidence-to-action bottleneck.

### Prompt Sensitivity: No Neutral-Analyst T2/T3

Location:

```text
results/prompt_sensitivity/
```

Key tables:

- `treatment_means_t2star_t3star.csv`
- `t2_t3_prompt_style_side_by_side.csv`
- `action_distribution_no_style_minus_neutral_deltas.csv`

T2*/T3* are upstream-summary variants that remove the neutral-analyst framing
from T2/T3. The intervention happens at the upstream summary generation stage,
before downstream decision agents receive the T2*/T3* representations. The key
ablation is the removal of the neutral-analyst role/framing, not a generic
change in writing style. These should be interpreted as robustness variants of
T2/T3, not as a separate baseline family.

### Appendix Diagnostics

Location:

```text
results/appendix_diagnostics/
```

Key tables:

- `belief_mad_contrasts_event_bootstrap.csv`
- `category_diversity_contrasts_event_bootstrap.csv`
- `profile_separability_formal_event_bootstrap.csv`
- `reasoning_contrasts.csv`

These tables support formal or auxiliary diagnostics used in the appendix and
paper robustness discussion.

### Human Audit: Evidence And Representation Validity Checks

Location:

```text
results/human_audit/
```

Key files:

- `audit_protocol.csv`
- `canonical_evidence_audit_sheet.csv`
- `cross_treatment_representation_audit_sheet.csv`

The released human-audit subset has three parts:

1. a compact audit protocol describing the audit scopes, labels, released
   details, and the fact that no inter-annotator agreement claim is made;
2. a row-level canonical evidence audit covering 62 evidence units, with
   public source excerpts and judgments for source alignment, numeric fidelity,
   claim fidelity, quote fidelity, category fit, support-label fit, and
   post-event leakage;
3. a row-level cross-treatment representation audit covering 4 rendered
   treatment representations, with rendered text, cited source IDs, validity
   checks, and audit notes.

The row-level sheets are released directly rather than only as aggregate
summaries. Reviewer identity and internal pipeline path columns are omitted
from the public CSVs, but the substantive audit content is retained. No
inter-annotator agreement claim is made.

The audit supports the construct-validity claim that the released evidence
banks and treatment representations can be inspected against visible,
event-time public source material, and that post-event leakage and unsupported
material claims were explicitly checked.

## Outcome And Input Boundary

Model-facing artifacts are built from public event-time source materials and
derived treatments. They should not include:

- CAR_1_5 labels
- post-event price reactions
- ex post quality labels
- evaluation-only joins that reveal outcome direction

If evaluation outcomes are included in a later release stage, they should be
placed under an explicit `evaluation_only/` directory and documented separately
from model-facing inputs.

## Excluded Operational Artifacts

The following artifact classes are deliberately not part of this release map:

- raw provider batch status directories
- repair/rebatch/rescue scratch directories
- local caches
- empty or historical error shards
- operational logs
- full raw downstream provider traces without a trace manifest

These files may be useful for local reproducibility archaeology, but they are
not the clean artifact surface for understanding the paper.

## Rebuilding The Release Subset

From the repository root:

```bash
python3 scripts/build_reviewer_result_release.py
```

The builder copies the allowlisted result, calibration, and audit artifacts into
`release/results/` and regenerates `results/result_table_manifest.csv` with row
counts and checksums.
