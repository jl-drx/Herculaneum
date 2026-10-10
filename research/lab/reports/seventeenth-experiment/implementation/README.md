# Phase 17 diagnostic implementation

All three CPU producers completed once under the root's frozen research design and amendments. They use retained, pinned inputs; no model, checkpoint, training, threshold fitting, crop/depth search or network operation occurs in these numerical helpers. Previous experiments remain unchanged.

The actual interpreter was `/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python`, Python 3.12.14, with NumPy 2.3.5, SciPy 1.17.0, Pillow 12.3.0 and Matplotlib 3.10.8. The complete binary/module identities are in `runtime-supplement.json`; `ppm-runtime-supplement.json` links the third run. The original result receipts retain their actual commands and runtime observations. This interpreter is distinct from `/workspace/.venvs/herculaneum/bin/python` used by other phases.

Recorded commands, with working directory equal to this directory:

```sh
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python run_ct_diagnostic.py --plan /workspace/herculaneum-lab/reports/seventeenth-experiment/implementation/ct-plan.json --output-dir /workspace/herculaneum-lab/reports/seventeenth-experiment/implementation/ct-run
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python run_ir_diagnostic.py --plan /workspace/herculaneum-lab/reports/seventeenth-experiment/implementation/ir-plan.json --output-dir /workspace/herculaneum-lab/reports/seventeenth-experiment/implementation/ir-run
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python run_ppm_diagnostic.py --plan /workspace/herculaneum-lab/reports/seventeenth-experiment/implementation/ppm-plan.json --output-dir /workspace/herculaneum-lab/reports/seventeenth-experiment/implementation/ppm-run
```

`PYTHONDONTWRITEBYTECODE=1` was set. The first two shell invocations used `python`; their result receipts resolve that alias to the full executable above. Each helper fixes numerical thread settings to four, disables network connections, uses exclusive output creation and records resource limits. Each plan pins its scientific protocol, code and source metadata. Each completed run has `input-integrity.json` with all consumed file identities checked before and after use.

| Output | Contents |
|---|---|
| `ct-run/ct-diagnostics.json` | Six cases; all 51 native and 17 pooled depth profiles on four fixed supports; coordinate checks; bitwise pooling equality; six fixed CT panels |
| `ir-run/ir-diagnostics.json` | Exact original PNG canvases/modes; six crop-first IR gray/darkness/local-darkness arrays; tied-rank AUROC/AP; six IR panels |
| `ppm-run/ppm-diagnostics.json` | Both endian hypotheses for every one of six objects across both cases; all 3×3 sampled records; conditional geometry; same-point variant comparisons; fixed-site support coverage |

The CT producer streamed 812,187,648 useful uint16 sample bytes from 1,530 retained multipart bodies, without saving redundant native volumes. All 15,040,512 released float32 pooled voxels matched bitwise. Native support repeats each literal coarse mask pixel 3×3; it is not a new native segmentation. Native and pooled statistics preserve their different intensity scales. Empty groups and undefined geometry/rank statistics retain null values.

Infrared global and local darkness quantify agreement with labels derived from infrared, not independent optical registration or CT ink detection. The fixed local Gaussian is applied inside the native crop before 3×3 averaging.

PPM byte order is an invariant-consistent conditional interpretation, not authenticated historical endianness. Both candidates were checked against all 1,458 records per object, and little-endian uniquely passed for all six. Unmapped records invalidate relevant center differences; eight unmapped result-map centers fall outside literal surface, known and primary supports. Flat-map coordinates and result-map coordinates are different representations: their numerical difference is not a measured physical displacement. Normal signs do not establish anatomical inward/outward orientation.

Synthetic checks: 39 decoder/pooling/coordinate, 21 statistic/rank, 10 infrared schema/filter, 56 PPM invariant/geometry and 6 multipart assembly checks, plus CT/IR rendering smoke checks. `synthetic-evidence/archive.json` preserves byte-identical copies of temporary fixture receipts referenced by the original frozen plans. PPM fixture receipts are retained directly in this directory. Independent recomputation receipts are maintained separately under `../independent-review/`.
