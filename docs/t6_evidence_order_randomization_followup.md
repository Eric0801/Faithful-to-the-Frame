# T6 Evidence-Order Randomization Follow-up

## Scope

T6 is a post-review, four-model order/salience sensitivity control for the
canonical B0 evidence bank. It uses one deterministic, event-specific
randomized evidence-unit permutation for each of the 94 events. The manifest
is frozen before provider submission. This is a protocol specification, not a
formal public preregistration.

T6 is compared with the verified four-model B0 snapshot generated in the
immediately preceding T5 confirmation campaign. It does not replay B0: the
baseline payloads, prompt template, model-family IDs, profiles, seed policy,
and canonical evidence banks are unchanged, while T6 is submitted in the same
provider campaign window.

## Estimand

The intervention preserves the event metadata and full evidence-unit multiset.
It only permutes the top-level canonical evidence-unit sequence. It therefore
estimates sensitivity to the presentation order of an otherwise fixed,
contextualized canonical bank. It does not identify effects of issuer identity,
source selection, factual management claims, or source framing generally.

The primary contrast is the event-level difference in positive expected-return
rate: `T6_canonical_evidence_order_randomized - B0_canonical_evidence_only`.
The inference unit is the event; profile and model cells are aggregated within
event before a paired 20,000-replicate event-cluster bootstrap. Secondary
endpoints are expected-return magnitude, buy rate, directional accuracy, and
action accuracy. Quality endpoints are guardrails, not equivalence claims.

## Randomization and Audit Gates

For each event, a master seed and `event_id` derive a separate pseudorandom
seed. A Fisher-Yates permutation is rejection-sampled until non-identity and
then frozen. Thus there is one realized randomized ordering per event, not one
global ordering policy for all events.

Before submission, the audit requires exact 94-event coverage; identical unit
counts; identical `evidence_id` multisets; canonical-JSON equality of every
unit matched by ID; unchanged metadata serialization; and a changed sequence
for every event. The manifest records ordered and unordered hashes, original
and permuted IDs, per-event seed digest, and position mapping.

The intervention can disrupt discourse-level adjacency in the source-derived
bank. It is therefore reported as an order/salience control, not a clean
de-framing treatment or a test of raw issuer-document order.
