# Camera-ready controls

This directory records the post-review experimental additions used in the
camera-ready paper. It contains source code, protocols, and compact inference
readouts, but excludes provider request payloads, raw responses, batch status,
caches, and other operational artifacts.

## T5: linguistic de-framing

T5 is paired with the canonical evidence-only B0 treatment. It preserves every
evidence unit, metadata field, source ID, value, and ordering; it may only
remove pre-approved local evaluative or self-presentational wording in claim
text. The full protocol is [t5_linguistic_deframing_followup.md](t5_linguistic_deframing_followup.md).

The four-model inference is released in
`results/camera_ready_controls/t5_followup_summary.json`. The edit-bearing
robustness analysis is separately released in
`results/camera_ready_controls/t5_edit_bearing_inference.json`. This is a
narrow mechanism probe, not a general mitigation claim.

## T6: evidence-order randomization

T6 retains the B0 evidence-unit multiset and changes only a frozen,
event-specific permutation of top-level evidence-unit order. The full protocol
is [t6_evidence_order_randomization_followup.md](t6_evidence_order_randomization_followup.md).

The pooled four-model readout is released in
`results/camera_ready_controls/t5_t6_joint_pplus_summary.json`; the strict
action-accuracy guardrail/equivalence analysis is in
`results/camera_ready_controls/t6_action_accuracy_tost.json`. T6 is an
order/salience sensitivity control, not a general test of source framing.

## E1: entity-prior hierarchy validation

`code/scripts/validate_e1_prior_hierarchy_outputs.py` validates the recovered
four-arm E1 matrix against the rendered evidence-visible replay packets. It
checks arm completeness, prior-only citation absence, replay citation
visibility, model/profile coverage, and row counts. Its focused unit test is
`tests/test_validate_e1_prior_hierarchy_outputs.py`.

The recovered raw matrix and packets are not duplicated here because their
provenance copy remains outside the public artifact. The camera-ready
validation record is [e1_four_model_recovery_validation.md](e1_four_model_recovery_validation.md).
It reports the exact validation contract and outcome without claiming a
fresh-provider rerun.
