# Conditional final-state retention

This is the separately frozen inference stage of the thirteenth conservative CPU continuation. The scientific contract is [RESEARCH_DESIGN.md](../RESEARCH_DESIGN.md), SHA256 `59bee53d3f1a662b37aec43fde31698ce0a9946ea0dd0de7027c01c3cf254f36`. Plinius is a provisional series label for an adaptation of the credited upstream seed42 model; this stage makes no method novelty or independent accuracy claim.

Only final A_preserve and B_preserve states from the fixed 32-update, learning-rate `1e-6` continuation can enter inference. For each arm, the training gate takes float64 equal-batch mean risks over the same 32 schedule exposures, computes `base_final - base_initial` and `full_final - full_initial`, and requires both differences to be `<= -1e-4`. All four mean risks must be finite numeric values; stored deltas must exactly equal the stated subtraction, with no tolerance. The gate is bound to the worker's final raw-state record and semantic hash and is independently verified before root creates a ready retention plan. An arm that fails is recorded and receives no retention inference. If neither qualifies, the driver and analyzer emit `SKIPPED`, with zero new model calls and no substitute state.

The four canonical cases are inherited through existing JSON receipts:

| Contexts | Exact source and input | Forward depth | Fixed primary support |
|---|---|---|---|
| PHerc0814 `pherc0814-46527-fresh-rank01`, `...rank14` | Unchanged eighth acquisition, tenth diagnostic case objects; uint8 `[21,384,384]`, 9.596 µm pooled representation | `2:19`, original layer-start 2 | Supplied supervision ∩ observed acquisition ∩ 64-pixel crop interior; 49,033 and 21,328 participants |
| Frag4 `frag4-context1`, `frag4-context2` | Unchanged eleventh uint16 CT pooling; float32 `[17,384,384]`, 9.72 µm representation, exact `float32(uint32_sum27 / float64(27*257))` | Entire 17 planes, original layer-start 0 | Original known mask ∩ 32-pixel crop interior |

Inference uses the unchanged retained reference adapter and robust MAD 1/99 normalization of each `[17,128,128]` tile. No labels, surface masks or reference participation enter normalization. Each context has the original 25 tiles, stride 64, unchanged sigmoid blending and forward depth direction. There is no new reverse/depth/offset search, resampling, fitted contrast or inference on calibration regions. Every declared context is retained for every eligible state. Fixed primary masks must remain exactly equal to their canonical masks and fully evaluated/finite; failure stops instead of shrinking support.

The sixth strict model-contract helper instantiates the original CPU factory and validates the pinned initializer. Eligible raw state dictionaries load with `weights_only=True`, exact tensor keys/shapes/dtypes/finite values, factory storage aliases and strict reload. The final semantic hash must match both the training gate and worker result. The archived eleventh Capture wrapper stores original tile logits and input fingerprints without changing the model inputs or outputs. Frag4 fingerprints must exactly match its existing forward baseline record. Historical PHerc0814 results did not cache normalized tile fingerprints: their comparison is bounded by unchanged source, mask, layer, normalizer and adapter pins and exact tile placements; no independent historical fingerprint equality is invented. Final model state and absence of parameter gradients are checked after inference.

Primary metrics are tied-score ROC AUC and grouped average precision on each unchanged fixed primary mask. Reuse the frozen seed42 raw maps and matched old A_preserve/B_preserve `1e-4` raw maps from the tenth PHerc0814 and eleventh forward Frag4 runs. These retained states are cross-checked by semantic hashes across both studies. New states are reported against baseline and their own old continuation. No baseline forward occurs. Secondary confusion/sensitivity/specificity use the common retained baseline cutoff **0.7558947205543518** with `float64(score) >= cutoff` for every role; old per-role cutoffs are not used and no threshold is fitted.

Report each context, the equal-context macro separately for PHerc0814 and PHercParis1Fr39, and descriptive equal-context/equal-specimen summaries. Four contexts represent two specimens, with repeated/overlapping sampling history; no pixel confidence intervals or independent sample claim. The descriptive screen requires each domain macro AUC/AP delta from baseline `>= -0.01` and each context AUC/AP delta `>= -0.02`. These exact absolute-difference checks do not select arms, trigger retries or automatically promote a state.

Limits are at most two sequential eligible workers, 100 tile forwards per worker and 200 total; four CPU threads, 120 seconds and 360 CPU seconds per worker, 300 seconds combined inference, 3 GiB RSS acceptance, 12 GiB address-space ceiling, and 64 MiB combined inference-plus-analysis artifacts. The separate analyzer is capped at 120 wall/CPU seconds, 1 GiB RSS acceptance and 4 GiB address space. Pin Python 3.12.14, NumPy 2.5.3 and torch 2.12.0+cpu. Socket connection traps enforce offline execution, and there are zero automatic retries, checkpoint outputs, training calls or threshold-fit calls. Keep fresh output directories and preserve all earlier files byte-for-byte.

Preparation reads only saved JSON and source text. The pending plan has no executable checkpoint binding. Root can create the ready plan only after training, independent gate verification, final technical review and source seal:

```sh
/workspace/.venvs/herculaneum/bin/python prepare_retention.py --output-dir pending-plan
/workspace/.venvs/herculaneum/bin/python prepare_retention.py --output-dir final-plan --gate ../implementation/analysis/retention-gate.json --root-decision ../RETENTION_DECISION.md --review PATH_TO_INDEPENDENT_TRAINING_VERIFICATION --review PATH_TO_RETENTION_GO --ready
/workspace/.venvs/herculaneum/bin/python run_retention.py --plan final-plan/retention-plan.json --output-dir run
/workspace/.venvs/herculaneum/bin/python analyze_retention.py --plan final-plan/retention-plan.json --run-dir run --output-dir analysis
```

The PHerc0814 cases and Frag4 are reused development/retention evidence with unresolved historical model exposure; Frag4's optical registration and annotation qualifications remain. No authenticated ink, reading, translation, prospective holdout, external accuracy, automatic deployment or paid-compute claim follows from this stage.
