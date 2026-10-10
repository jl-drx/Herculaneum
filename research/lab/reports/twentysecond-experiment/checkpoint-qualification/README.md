# Native artifact qualification, Stage 1

Prepared only. A written root GO binding this exact plan and code-freeze SHA256 is required before the CLI can execute. No native checkpoint values, network requests or models were used during preparation.

`qualify_checkpoint.py` has two separate components: a pinned exact-range downloader with durable pre-open records and bounded manual redirects, and a format-aware offline F32/alias inspector. The parent enforces a 240-second total wall limit, samples RSS every50ms, and retains all partial output. Kernel address-space and offline CPU limits supplement that monitor. HTTP200, wrong validators, malformed format, nonfinite values and unequal aliases stop without retry. URL queries, redirect bodies and raw transport exception messages are never retained. The only completed download is renamed after its full SHA256 matches; no checkpoint conversion is created.

The offline inspector rehashes the complete file, requires the retained prefix/header byte-for-byte, validates all508 descriptors and contiguous offsets, scans every named F32 value and compares every declared alias pair byte-for-byte. A PASS establishes artifact integrity/value consistency; it does not establish strict loading, runtime parity, accuracy or transfer. Conditional factory storage-group counts are not shared serialized-storage claims.

Preparation tests use tiny authored F32 files and fake transports. `synthetic-tests.json` retains the exact fixture outcomes. A separate reviewer must inspect this code/plan and prepare independent safetensors-based acceptance before root execution authorization.

Authorized execution shape (root supplies a fresh output path and GO):

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /workspace/.venvs/herculaneum/bin/python -B qualify_checkpoint.py \
  --plan PLAN.json --freeze CODE_FREEZE.json --go ROOT_GO.json --output run-v1
```

The GO must be JSON with `status: "GO"`, `plan_sha256` and `code_freeze_sha256`. Both preparation inputs and every frozen code/test/report file are rechecked before execution. Existing output directories are rejected. The controller writes a final sanitized receipt even after an external wall/RSS stop. No resume, retry, source substitution or Stage2 model load is implicit.
