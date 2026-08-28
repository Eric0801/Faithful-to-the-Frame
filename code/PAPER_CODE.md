# Paper code map

This list is the reproducibility-code surface for the EMNLP paper. Scripts retain their original filenames so documented commands and sibling imports remain valid.

| Script | Stage | Purpose |
| --- | --- | --- |
| `scripts/find_matching_events.py` | data_construction | SEC event candidate collection |
| `scripts/select_clean_strict_reservoir.py` | data_construction | stratified main-sample selection |
| `scripts/build_source_packets.py` | data_construction | event-time source packet construction |
| `scripts/build_entity_visible_packets.py` | data_construction | entity-visible packet rendering |
| `scripts/build_evidence_banks.py` | data_construction | canonical evidence-bank construction |
| `scripts/build_hidden_outcomes.py` | data_construction | evaluation-only CAR label construction |
| `scripts/render_treatments.py` | treatment_generation | T1/T2/T3/T4 rendering |
| `scripts/render_b0_canonical_evidence_only.py` | treatment_generation | B0 canonical evidence-only rendering |
| `scripts/render_t4_full_ledger.py` | treatment_generation | T4 structured-ledger rendering |
| `scripts/render_t5_linguistic_deframing.py` | treatment_generation | T5 local lexical control rendering |
| `scripts/build_t5_linguistic_deframing_manifest.py` | treatment_generation | T5 candidate manifest |
| `scripts/apply_t5_linguistic_deframing_adjudications.py` | treatment_generation | T5 approved-edit application |
| `scripts/freeze_t5_linguistic_deframing_manifest.py` | treatment_generation | T5 manifest freeze |
| `scripts/build_t6_evidence_order_randomization.py` | treatment_generation | T6 fixed-content order randomization |
| `scripts/freeze_t6_evidence_order_manifest.py` | treatment_generation | T6 permutation-manifest freeze and integrity record |
| `scripts/build_downstream_requests.py` | execution | downstream request generation |
| `scripts/build_b0_canonical_baseline_requests.py` | execution | B0 request generation |
| `scripts/build_t5_linguistic_deframing_requests.py` | execution | T5 request generation |
| `scripts/build_t6_evidence_order_requests.py` | execution | T6 request generation |
| `scripts/experiment_run_profiles.py` | execution | fixed receiver-profile definitions |
| `scripts/harness_openai_compat.py` | execution | OpenAI-compatible provider client |
| `scripts/run_representation_harness.py` | execution | upstream representation harness |
| `scripts/run_downstream_harness.py` | execution | downstream decision harness |
| `scripts/validate_decision_outputs.py` | validation | schema and source-ID validation |
| `scripts/audit_outcome_blindness.py` | validation | outcome-blindness audit |
| `scripts/audit_masking_leakage.py` | validation | masked-packet leakage audit |
| `scripts/repair_representation_source_refs.py` | validation | deterministic source-reference repair |
| `scripts/validate_e1_prior_hierarchy_outputs.py` | validation | E1 four-arm packet/row validator |
| `scripts/compute_diversity_metrics.py` | analysis | standard diversity and quality metrics |
| `scripts/compute_reasoning_semantic_diversity.py` | analysis | semantic reasoning diagnostics |
| `scripts/compute_upstream_compression_metrics.py` | analysis | upstream T2/T3 compression diagnostics |
| `scripts/compare_t2_t3_style_runs.py` | analysis | neutral-style versus no-style comparison |
| `scripts/bootstrap_b0_event_level_contrasts.py` | analysis | B0 event-cluster bootstrap |
| `scripts/build_b0_canonical_baseline_readout.py` | analysis | B0 readout tables |
| `scripts/build_action_behavioral_convergence_20260507.py` | analysis | action convergence diagnostics |
| `scripts/build_paper_missing_metric_readout_20260517.py` | analysis | paper metric/bootstrap tables |
| `scripts/analyze_t5_edit_bearing_inference.py` | analysis | T5 edit-bearing inference |
| `scripts/analyze_t5_followup_results.py` | analysis | T5 full and edit-bearing summary |
| `scripts/build_rebuttal_w3_conditional_positive_predictions_20260712.py` | analysis | conditional positive-prediction readout |
