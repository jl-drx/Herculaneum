The original audit stopped because its 160 MiB guard counted every state-dictionary name, including duplicate names for the same storage. Its checkpoint-step guard also assumed the filename's 75000 was the stored integer. The original code, plan, failed audit and preparation record remain unchanged.

An additive offline diagnosis loaded all four SHA-verified checkpoints using `weights_only=True`, instantiated the unchanged factory, and performed strict state loading without inference. Every arm has the same measured contract:

| Measurement | Checkpoint and factory |
| --- | ---: |
| Named tensors | 508 |
| Named tensor bytes | 272701704 |
| Unique storages | 264 |
| Unique storage bytes | 138185992 |
| Exact duplicate-view alias groups | 244 |
| Duplicate named bytes | 134515712 |
| Unique parameters | 34546498 |
| Stored `step` | 74999 |

All complete key sets, shapes and dtypes match. Storage alias partitions exactly match the factory, and every expected alias group's checkpoint values are bitwise equal. Strict factory loading succeeds in all four arms. The aliases include `all_modules.0.weight`/`conv.weight` and `all_modules.1.weight`/`norm.weight` paths to the same tensors; no keys were discarded or rewritten. This establishes local technical correspondence, not a scientific accuracy or ink result.

The diagnostic script is `diagnose_storage.py`; evidence is `storage-diagnostic/storage-diagnostic.json` (SHA256 `c89e57e401c02777947a33bb19359738d40db129392121e4ae894c73d3f57288`). It verifies the original frozen package and failed-audit hashes before and after, blocks network sockets, uses four CPU threads and the same 120 CPU/180 wall seconds and 8 GiB process limits. It records zero network requests and zero inference calls. Actual Russo embedded configs also match every published author field and the intended six-label-root pair; their supervision/dynamic-label overrides are absent.

The additive `model_contract_v2.py` keeps the 160 MiB bound for unique storage, and additionally requires the exact observed counts, complete factory key/shape/dtype schema, storage byte allocation, stride/offset layout, alias partition and bitwise alias equality. It retains original archive/tensor/parameter/CPU bounds, safe loading and final file SHA verification. It pins `embedded_step=74999` in each artifact record and reports `filename_step_label=75000` separately. Historical trainer indexing convention is unproved; no such convention is inferred from filenames. Public model revisions, object hashes and downloaded bytes are unchanged.

V2 preparation rechecked all 28 pinned inputs and the four-arm diagnostic evidence, syntax and actual/published config correspondence. It did not rerun the original audit, perform v2 checkpoint loads or infer. Parent review/freeze and separate execution authorization are required before this offline audit command:

```sh
/workspace/.venvs/herculaneum/bin/python /workspace/herculaneum-lab/reports/sixth-experiment/model-contract/audit_checkpoints_v2.py \
  --plan /workspace/herculaneum-lab/reports/sixth-experiment/model-contract/model-plan-v2.json \
  --plan-sha256 f746d1bb94ce946a57ff9a9f62adf2f7ca9021d547db7ef541a301e0c99e8cb6 \
  --output-dir /workspace/herculaneum-lab/reports/sixth-experiment/model-contract/audit-v2
```

The helper APIs and four role labels remain unchanged. A passing result is `audit-v2/model-contract.json`; the benchmark must use its pinned artifact records and `model_contract_v2.py`. The raw score and interpretation limits in the original plan continue to apply. No new checkpoint download, paid action, reading, transcription or translation is authorized by this revision.
