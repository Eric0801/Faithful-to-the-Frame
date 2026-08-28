# E1 four-model recovery validation

The E1 recovery check validates a four-arm entity-prior hierarchy matrix
against its exact evidence-visible replay packets. It is a validation record,
not a new provider generation.

The expected matrix has 9,024 decision rows: 94 events, six profiles, four
receiver-model families, and four arms. The three prior-only arms must report
no cited evidence; the evidence-visible replay must cite only source-unit or
structured-fact IDs visibly rendered in its event packet.

The recovered matrix passed this contract: all 9,024 rows were valid, with 94
events and 2,256 rows per arm. The checked model families are Claude Sonnet
4.5, GPT-5.2, Qwen3-235B-A22B, and DeepSeek-V3.1.

Run the validator against an authorized copy of the recovered decision table
and packet tree:

```bash
python3 code/scripts/validate_e1_prior_hierarchy_outputs.py \
  --input-csv E1_DECISION_ROWS.csv \
  --rendered-packets E1_RENDERED_PACKETS \
  --report reproduced/e1_validation.json \
  --expected-total 9024 --expected-events 94 --expected-profiles 6 \
  --expected-model-families claude-sonnet-4.5,gpt-5.2,qwen3-235b-a22b,deepseek-v3.1
```

The focused regression test is
`tests/test_validate_e1_prior_hierarchy_outputs.py`. The provenance-preserved
recovery copy is not duplicated in this public artifact.
