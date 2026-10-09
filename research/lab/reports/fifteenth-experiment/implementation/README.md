# Label-only benchmark construction

This package performs six fixed PNG downloads and three sequential offline label selectors. It accesses no CT, IR, models or checkpoints. The scientific source and benchmark contracts fix Frag1/Frag2 as Paris2 evaluation fragments and Frag3 as Paris1 calibration. Failed source/support gates produce an incomplete construction without substitutions or relaxation.

Author synthetic checks passed20 cases; independent technical review passed27 cases and rechecked the20 author cases. Only synthetic images/arrays and in-memory HTTP stubs were executed during preparation. `draft-plan-v1/label-plan.json` is not ready for acquisition. The source-bound technical GO is `../review/code-go.json`.

Root creates a new ready plan after binding the independent checker preparation receipt. The builder requires a top-level `status: GO` JSON with file records binding all six exact source files. Additional `--review` JSON records are pinned; use the checker receipt path once sealed.

```bash
cd /workspace/herculaneum-lab/reports/fifteenth-experiment/implementation
/workspace/.venvs/herculaneum/bin/python prepare_plan.py \
  --output-dir final-plan --ready \
  --review ../review/code-go.json \
  --review fixtures-v1/fixture-receipt.json \
  --review code-freeze.json \
  --review ../runtime-metadata.json
/usr/bin/timeout 180 /workspace/.venvs/herculaneum/bin/python acquire_labels.py \
  --plan final-plan/label-plan.json --output-dir acquisition
/usr/bin/timeout 420 /workspace/.venvs/herculaneum/bin/python run_selection.py \
  --plan final-plan/label-plan.json \
  --acquisition acquisition/provenance.json --output-dir selection
```

Append the independent checker preparation JSON as another `--review` argument before ready-plan creation. Root alone runs the real downloads and selections. All output directories must be fresh. Stop on any failure and preserve receipts; do not rerun the scientific stage.

The acquisition ledger distinguishes retained body bytes from conservative failed-attempt charges. Successful objects require HTTP200, identity encoding, complete PNG structure/CRC and declared canvas; unknown-length responses must end before their cap. Selection starts only after all six complete, hashed objects exist. Modes1/L/P accept Boolean,0/1 or0/255 membership, without appearance-based polarity changes; used palette entries must have black0 and white nonzero values. Transparency is rejected.

`selection/panel-provenance.json` aggregates three worker reports and six contexts. Each worker report retains all candidate counts, exact90% gates, quality/palette facts, outside-support positives and the selected source coordinates/roles. Each selected context has `labels.npy` (uint8), `surface-mask.npy`, `known-mask.npy` and `primary-mask.npy` (bool), all384×384:24 hashed arrays in total. Outside-support positives are recorded without becoming an extra selection gate. Unknown pixels remain excluded from participation.

The common method is the retained eleventh symmetric33×33 exclusion, native1152-grid, coarse centers3i+1, border32, positive512/negative8192 gates and first/last eligible Y separation2304. Worker limits are120 wall seconds,360 CPU seconds,4 threads,3GiB peak-RSS acceptance and12GiB address space. Aggregate selection has420 seconds; acquisition has180 seconds. Combined acquisition/selection output is capped at8MiB. Runtime versions are pinned; there is no package installation, network in selection, image prediction or calibration fitting.

`planning/ct-cost-estimate.json` and `planning/CT_COST.md` are metadata-only future-cost arithmetic. They do not authorize CT acquisition or establish TIFF/range readiness. Legacy license restrictions, uncertain upstream exposure and unmeasured registration error remain attached to the root contracts.
