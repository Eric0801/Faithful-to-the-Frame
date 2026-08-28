# T5 Linguistic De-Framing Follow-Up

Status: execution specification for the reviewer-requested paired control.

## Estimand

Conditional on the entity-visible B0 canonical evidence bank, what is the
effect of removing pre-specified evaluative and self-presentational linguistic
framing on downstream LLM-agent traces?

This is a linguistic intervention only. It does not identify effects of source
selection, source ordering, category grouping, source provenance, or entity
priors.

## Treatment Pair

`B0_canonical_evidence_only` is the source-faithful canonical-bank anchor.
`T5_linguistic_deframing` uses the same event metadata, canonical evidence
units, evidence IDs, category labels, values, source IDs, support labels,
unit order, JSONL serialization, receiver prompt, profiles, and decision seed.
Only an approved local edit to the `claim` field may differ.

## Allowed Frame Labels

- `self_evaluation`
- `promotional_superlative`
- `rhetorical_intensifier`
- `speaker_stance_or_self_credit`

The treatment uses conservative span-level edits. It must not use a neutral
analyst-style prompt, create a summary, aggregate across units, introduce a
recommendation, or add an inference.

## Protected Content

The following content must be preserved exactly unless the entire source unit
contains no operational proposition and is explicitly adjudicated:

- numeric values, units, dates, and periods
- increase/decrease direction, negation, and comparison baselines
- guidance, risks, uncertainty, conditions, and caveats
- entity and segment references
- evidence IDs, categories, values, source IDs, and support labels

## Audit Gates

1. Every manifest row matches one canonical `event_id` plus `evidence_id`.
2. All non-claim fields have exact parity with the canonical bank.
3. Each altered claim has an edit label and a human review status.
4. The renderer reports changed-row count, character/token deltas, and source
   coverage parity.
5. A stratified audit checks factual parity and framing removal before provider
   submission.

## Protocol-Specified Analysis Plan

This contrast set and analysis plan are fixed before any official T5 or B0
replay outputs are collected. This is protocol specification, not a formal
public preregistration.

Let `E` be the event set with at least one approved claim edit in the frozen
manifest. The following are co-primary estimands, reported with separate
event-cluster bootstrap confidence intervals rather than selected after
observing outcomes:

1. **Dataset-wide policy effect:** `T5 - B0` over all 94 events. This estimates
   the average effect of applying the specified linguistic de-framing policy to
   the complete study population, including events for which the policy makes
   no edit.
2. **Edit-bearing event effect:** `T5 - B0` restricted to `E`. This estimates
   the conditional effect where the pre-specified linguistic intervention was
   actually applied.

`E` is determined only by approved manifest edits before provider submission;
it cannot be changed after T5 outcomes are observed. We use `edit-bearing`
rather than `per-protocol`, because the latter implies treatment non-adherence
that does not occur in this deterministic transformation.

Before submission, the reviewed manifest is frozen in a machine-readable
record containing its SHA-256 hash, UTC timestamp, git revision, approved-edit
count, and the exact event IDs in `E`. Batch preparation and submission verify
that record against the manifest and declared counts; a changed manifest cannot
silently reuse the old freeze record.

The provisional dual-review set contains 13 proposed edits in 11 unique events.
The final `|E|` is frozen with the reviewed manifest. With a subgroup this
small, its interval is necessarily precision-limited: it can provide targeted
mechanism evidence, but cannot support a strong null claim or broad mitigation
claim when its interval is wide. Both estimands remain reported even if one is
near zero; the full-sample contrast must not be treated as evidence against an
edit-bearing effect merely because it is mechanically diluted.

## Confirmatory Run

The confirmatory run includes a contemporaneous B0 replay as well as T5. This
avoids attributing model-version or provider-time variation to a linguistic
edit when a T5 output is compared with a historical B0 output.

Primary receiver matrix: 94 events x 6 profiles x 2 treatments (B0 replay,
T5) x 4 model families (`gpt-5.2`, `claude-sonnet-4.5`,
`qwen3-235b-a22b`, `deepseek-v3.1`) x one fixed decision seed = 4,512 paired
traces.

The follow-up matches the original four-family robustness matrix, enabling
direct per-family robustness readouts alongside the pooled event-level
estimand. One fixed decision seed is used to limit spend, so the probe does
not estimate downstream seed variability at the scale of the original study.

The primary contrast is `T5_linguistic_deframing - B0_canonical_evidence_only`.
Inference resamples at the event level. The edit-bearing analysis additionally
reports an exact paired sign test and an exact paired sign-flip permutation
test on the 11 event-level deltas as robustness checks for the small-cluster
bootstrap interval. Quality is a pre-specified guardrail; non-rejection of a
quality difference is not evidence of non-degradation.

## Pre-Result Interpretation

For the edit-bearing estimand, `P+` is the event-level mean proportion of
downstream traces with positive `expected_return_5d`. If `P+` is directionally
lower under T5 than contemporaneous B0, this is targeted mechanism evidence
that the four specified lexical categories contribute to propagated skew. If
the interval spans zero or the direction is not consistent, it indicates that
the propagated skew in this sample is not primarily carried by those local
evaluative spans; source selection, ordering, or comparison-class framing
remain plausible channels outside T5's estimand. Neither outcome is evidence
against the separately reported dataset-wide policy effect.
