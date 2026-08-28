# Camera-ready control readouts

These small JSON files are paper-facing aggregate inference outputs. They are
not raw decision traces and do not contain provider request payloads or batch
status records.

| File | Contents |
| --- | --- |
| `t5_followup_summary.json` | Four-model, event-cluster T5-minus-B0 inference, including the pre-specified full and edit-bearing scopes. |
| `t5_edit_bearing_inference.json` | Exact-sign and paired sign-flip robustness readout for the frozen edit-bearing subset. |
| `t5_t6_joint_pplus_summary.json` | Pooled four-model T5/B0/T6 positive-expected-return comparison. |
| `t6_action_accuracy_tost.json` | T6 strict action-accuracy guardrail/equivalence analysis. |

The T5 and T6 protocols are in `docs/`. Reproduction scripts are listed in
`code/PAPER_CODE.md`.
