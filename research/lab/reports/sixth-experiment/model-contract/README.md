This package prepares two exact Russo 75k checkpoint objects for a four-arm w029 study. Preparation used retained public metadata and author configs; it read no checkpoint bytes, made no new network requests, and performed no model inference or paid action. Retrieval still requires the parent's separate execution authorization.

The immutable Hugging Face revision is `3e5e7ed44e04606cf4a3766cf83a016543a30d83` in `domenicor046/ink9um-dense`:

| Role | Object | Bytes | LFS SHA256 |
| --- | --- | ---: | --- |
| russo_ex01675k | dense9um-w016excluded-step075000.pth | 138367629 | 1d7b99bc641b459736a762b2a1d8c357f61ada4f0117c24c7327b6a1293ba1eb |
| russo_control75k | control-manuallabels-step075000.pth | 138366975 | e60cee6b2c08a021ddfa9683869dab91787531cef762d833cc1f21fdee7fa114 |

The new payload ceiling is 276734604 bytes, plus 128 KiB for redirect/error response bodies. There are at most twelve requests including five redirects per object, no automatic retry, and a five-minute wall limit per object. TLS remains enabled. The downloader does not load netrc credentials or HF tokens and records no signed redirect URL/query. An incomplete transfer stays as `.partial` and stops future automatic transfers. An existing complete object is reused only after exact size/SHA verification; a corrupt object stops. Atomic publication never overwrites a cached object. Free disk must retain at least 512 MiB beyond the object about to be transferred.

Run these only after the parent authorizes retrieval, using fresh output directories:

```sh
/workspace/.venvs/herculaneum/bin/python /workspace/herculaneum-lab/reports/sixth-experiment/model-contract/fetch_checkpoints.py \
  --plan /workspace/herculaneum-lab/reports/sixth-experiment/model-contract/model-plan.json \
  --plan-sha256 c53aea94800adee7a7594569ec06e33884ea73883eebf9fba031768511b3178d \
  --output-dir /workspace/herculaneum-lab/reports/sixth-experiment/model-contract/download

/workspace/.venvs/herculaneum/bin/python /workspace/herculaneum-lab/reports/sixth-experiment/model-contract/audit_checkpoints.py \
  --plan /workspace/herculaneum-lab/reports/sixth-experiment/model-contract/model-plan.json \
  --plan-sha256 c53aea94800adee7a7594569ec06e33884ea73883eebf9fba031768511b3178d \
  --output-dir /workspace/herculaneum-lab/reports/sixth-experiment/model-contract/audit
```

The offline audit blocks socket access before model imports. Its fresh subprocess permits four CPU threads, 120 CPU seconds, 180 wall seconds, and 8 GiB address space. It loads one model at a time with `torch.load(map_location='cpu', weights_only=True)` and performs no forward call. Expanded ZIP bytes, tensor bytes/count, parameter count, finite values, complete keys, shapes and dtypes are bounded and checked. No unsafe pickle fallback, additional safe globals, partial state load, implicit dtype repair, pretrained download, EMA substitution, depth padding or model substitution is allowed. All four roles must be step75000 and match the retained 34546498-parameter architecture and exact tensor schema.

`model_contract.py` exposes `load_runtime() -> dict` and `build_model(path, artifact_record, runtime) -> (model, metadata)`. The runtime uses the verified unchanged `run_ink_reference.py` factory, normalizer and inference function. The caller must set CPU limits and call `runtime['reference'].infer_surface` with its explicit study settings. Importing the helper performs no network, model loading or inference. `model-plan.json` contains the four artifact records (`label`, `repository`, `revision`, `repository_path`, `size_bytes`, `sha256`, `local_path`, `source_url`, `acquisition`) and pins every source/code input. A passing audit emits `audit/model-contract.json` with `models` keyed by those four labels, full tensor schemas and embedded configs, complete baseline differences and author-config checks. Failure preserves logs and emits `audit-execution.json` with stopped status; no automatic repair or rerun follows.

Published manual-control and ex016 configs agree on 29 representations and all global training settings except output directory. Exactly six label roots differ; w016 retains manual labels. The audit requires the same correspondence in the actual embedded configs, and checks every published config field. Unexpected differences stop for diagnosis. Matching config paths does not prove historical data/label bytes or historical training-code identity. The retained seed42/43 embedded configs omit `force_full_supervision`/dynamic-label overrides; current loader defaults remain conditional evidence about historical exposure.

Raw sigmoid values follow BCE training with 0.5 label smoothing; they are descriptive scores rather than calibrated ink probabilities. Shared training data and seed variants do not establish independent observations. This technical contract cannot establish ink authenticity, papyrus-layer identity, a reading, transcription or translation. The w029 benchmark's data masks/exposure contract and scientific comparison are separate. All reading fields remain null and no paid compute is authorized. Author scripts/docs are MIT; retained source licenses and checkpoint derivation/source terms remain applicable as recorded in the pinned model card and Russo LICENSE.

`preparation-checks.json` records successful syntax checks, all 17 retained input hashes, published pair validation, negative config guards, bounded error-body EOF/cap cases, redirect allowlist checks, exception redaction and offline socket traps. The actual HTTP transfer and model resource limits have not been exercised during preparation.
