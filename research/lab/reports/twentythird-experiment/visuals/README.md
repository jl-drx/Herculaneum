# Fixed phase23 context panels

Preparation only. No arrays, checkpoints, model outputs, networks or plotting imports were accessed during preparation. The code was parsed with Python's AST parser; rendering has not run. Native descriptor schema was confirmed with the implementation agent: `result.json.runs[case_id]` has `id` and `arrays.raw_scores` / `arrays.evaluated_mask`, each with `path`, `sha256`, `size_bytes`, `shape`, `dtype` and `raw_c_order_sha256`.

After root separately releases array plotting, `plot_contexts.py` renders exactly two PNGs: Paris1 with two complete contexts and Paris2 with four, in protocol order. Each row shows the supplied 0/1 reference, retained seed42 raw score, native raw score and native-minus-seed42 difference. Each image covers the complete 384² canvas. All panels use the original primary mask; unsupported pixels are gray. Probability scales are fixed to 0..1 and differences to −1..1. Labels have a discrete 0/1 legend. There is no subcropping, fitted threshold, contrast adjustment, normalization or model execution.

The plotter checks the protocol and final input-contract hashes, every source-file/raw-array hash, dimensions/dtypes, original support counts, full evaluated coverage and finite raw scores in [0,1]. It verifies the source hashes again after plotting. Its output receipt binds all 36 consumed arrays, the native result, input contract, protocol, code and both PNGs. It does not load CT surfaces, tile logits or model state. Combined PNG/receipt output is capped at 16 MiB.

The intended command, only after root release, is:

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python \
  /workspace/herculaneum-lab/reports/twentythird-experiment/visuals/plot_contexts.py \
  --native-result /workspace/herculaneum-lab/reports/twentythird-experiment/implementation/run-v1/native/result.json \
  --native-result-sha256 ACTUAL_VERIFIED_RESULT_SHA256 \
  --output-dir /workspace/herculaneum-lab/reports/twentythird-experiment/visuals/render-v1 \
  --root-release
```

Output names are `paris1-whole-contexts.png`, `paris2-whole-contexts.png` and `PLOT_RECEIPT.json`. Root performs visual review and scientific interpretation. The panels are supplied-reference/reused-development diagnostics, not independent accuracy evidence or transcriptions.
