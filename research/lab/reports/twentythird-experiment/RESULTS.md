# Plinius phase23: native reference helps Paris2 but regresses on Paris1

The fixed native-reference comparison completed and passed independent numerical verification. **The native checkpoint is not a consistently stronger replacement for seed42 on these inputs.** Paris2 improves, while Paris1 regresses, particularly frag3-context2. Retain seed42 as the adopted baseline and keep the native checkpoint as a contrasting research reference. No model promotion, training or GPU rental follows from this result.

These are descriptive supplied-reference results on six repeatedly used development contexts from two physical parents. They establish checkpoint behavior under a matched input contract, not independent accuracy, a new Plinius model, or the causal benefit of native pseudo-label training. No new ancient text was transcribed or translated.

## Matched comparison and results

Both checkpoints receive the same retained17-plane FP32 tensors, forward order, robust-MAD normalization,25-tile schedules per context and original primary support. The baseline is the unchanged seed42 checkpoint from `scrollprize/ink_9um`; the comparison is Nieuwlaar's published `ink9um-dense-native` checkpoint qualified in phase22. Their training histories differ; identical inference inputs do not turn this into a controlled training-method ablation.

The input audit independently matched all150 normalized tile hashes and reconstructed all six seed42 mosaics bit-for-bit from saved logits, so no baseline replay was needed. Native inference checked those hashes again before each forward. All520,794 original primary pixels were retained:86,535 supplied positives and434,259 negatives. The primary mask remains the inherited known mask intersected with the central320×320 area; unknown boundaries and the context border were not silently relabeled or dropped based on predictions.

Primary measurements use raw FP32 sigmoid scores blended in probability space. Parent summaries give equal weight to the prespecified contexts within each parent; no pooled-pixel significance test or overall average hides the opposing parent outcomes.

| Parent | Contexts | AUC: seed42 → native | ΔAUC | AP: seed42 → native | ΔAP |
|---|---:|---:|---:|---:|---:|
| Paris1 | 2 | 0.784788 → 0.630308 | -0.154479 | 0.240483 → 0.203584 | -0.036899 |
| Paris2 | 4 | 0.657642 → 0.705446 | +0.047804 | 0.384286 → 0.487602 | +0.103316 |

| Context | Parent | Seed42 AUC | Native AUC | ΔAUC | Seed42 AP | Native AP | ΔAP |
|---|---|---:|---:|---:|---:|---:|---:|
| frag3-context1 | Paris1 | 0.870064 | 0.848321 | -0.021743 | 0.298078 | 0.357457 | +0.059380 |
| frag3-context2 | Paris1 | 0.699512 | 0.412296 | -0.287216 | 0.182887 | 0.049710 | -0.133177 |
| frag1-context1 | Paris2 | 0.446052 | 0.430881 | -0.015171 | 0.185973 | 0.178302 | -0.007672 |
| frag1-context2 | Paris2 | 0.903678 | 0.907858 | +0.004180 | 0.500096 | 0.726190 | +0.226094 |
| frag2-context1 | Paris2 | 0.501298 | 0.627642 | +0.126344 | 0.269839 | 0.325355 | +0.055517 |
| frag2-context2 | Paris2 | 0.779541 | 0.855403 | +0.075862 | 0.581235 | 0.720561 | +0.139326 |

The native checkpoint improves AP on four of six contexts and AUC on three. Both measures improve on three contexts, both decline on two, and frag3-context1 trades higher AP for lower AUC. Even contexts from the same fragment differ. The largest loss is frag3-context2: AP0.182887→0.049710 and AUC0.699512→0.412296. Conversely, frag1-context2 gains AP0.500096→0.726190. Choosing a checkpoint separately for each known outcome would overfit this development evidence.

There was no prespecified performance pass/fail gate, and none was invented afterward. Execution and verification passed; the scientific outcome is mixed transfer. A threshold change cannot repair a poor raw ranking, and this study fitted no threshold, ensemble or selection gate.

## Spatial and quantization checks

Root inspected all six full-context panels with shared display scales and the unchanged support: [Paris1](visuals/render-v1/paris1-whole-contexts.png) and [Paris2](visuals/render-v1/paris2-whole-contexts.png). In frag3-context2 the native map weakens several lower supplied ink strokes while elevating broad background regions, consistent with its ranking regression. In frag1-context2 and frag2-context2 it more strongly follows portions of supplied strokes, although off-stroke responses remain. Both maps struggle on frag1-context1. These qualitative observations do not identify physical causes or authenticate the supplied labels. Gray regions are inherited support exclusions, not predictions of uncertainty.

The full [visual review](ROOT_VISUAL_REVIEW.json) records every context, including unfavorable cases. Plotting changed no data, crop, support or score. Six contexts are not six independent parent replications, and recognizable shapes in diagnostic panels are not a new transcription.

The frozen analysis separately measured uint8 truncation and the author's clipped `(uint8−64)/128` representation for BOTH models. They reduce many distinct raw scores to ties and must not be treated as calibrated probabilities. For example, native frag2-context2 AP changes0.720561 raw→0.716806 uint8→0.704208 decoded; frag2-context1 changes0.325355→0.325690→0.309561. The small non-monotonic AP effect of discretization reflects grouped ties; no performance correction was selected. Primary conclusions use the raw scores. All representation metrics, endpoint fractions, unique-score counts and paired probability differences are retained in [metrics.json](implementation/analysis-v1/metrics.json).

## Verification and compute

One fresh offline native worker performed exactly one construction, one strict safetensors state load and150 full-model forwards. It verified508 tensor values/hashes,244 alias pairs and264 factory groups, with34,546,498 unique parameters, before and after. All state and source bytes remained unchanged, with no gradient, unsafe checkpoint load or network call. The existing native file was reused without copying or conversion.

Runtime was Python3.12.14,NumPy2.5.3,Torch2.12.0+cpu,safetensors0.8.0;FP32 evaluation/inference,deterministic algorithms,four intra-op threads and one interop thread. Inference took15.447seconds under the external controller, with peak self-reported RSS838,111,232bytes (about799MiB). This is a local CPU measurement, not a GPU benchmark or historical CUDA reproduction. The saved-output analysis made no model call.

The independent frozen checker verified all150 input fingerprints, all loaded state hashes/aliases, exact original support and the full call count. Every raw/encoded/decoded metric, endpoint fraction and paired difference matches the author analysis within1e−12. Native probability sums,weights and raw mosaics reconstruct exactly; independent sigmoid/Hann equations differ by at most5.96e−8/1.79e−7, and independent seed42 blend arithmetic by at most2.98e−7, within frozen tolerances. It ran no model replay. See [independent verification](independent-review/RESULTS.md), [input reuse audit](input-audit/RESULTS.md) and [execution report](implementation/RESULTS.md).

Preparation included42 author toy checks,31 independent metric/support tests with63 small oracle cases,and11 independent evidence tests. One mislabeled synthetic metric fixture was corrected before native output; its original source/log remains retained. No actual scientific execution or checker retry occurred.

## Interpretation and next step

The checkpoint is technically usable but its regional tradeoffs rule out a universal replacement on the evidence available. The input, loading and arithmetic checks passed; the observed difference is checkpoint behavior under this fixed contract. The study does not establish whether orientation, render depth, source domain, annotation limitations or training history caused the failure. A fitted ensemble on these six outcomes would not resolve that uncertainty.

The [next study design](NEXT_STUDY_DRAFT.md) proposes one paired depth-direction diagnostic on all six contexts and both fixed models: reverse the existing17-plane order while preserving its values, support and all other processing. Reuse the forward outputs and prescribe300 new reverse-direction forwards before seeing them. Treat the result as a counterfactual sensitivity measurement, not permission to choose a best direction per region or claim historical normals were wrong. Root must freeze its execution and checker contracts separately. No extra experiment was run in this phase.

Keep PHerc0009B,PHerc0500P2,PHerc0841 and Frag6 reserved and unscored; phase22's unresolved ancestry/render/optical-label contracts remain. No GPU resource, credential handling, external upload, publication or prize submission occurred. GPU spend remains£0 and the£10TOTALfirstpilot cap persists, including failures/retries/overheads. Before any paid pilot, a useful frozen workload,pinned CUDA,current pricing,independent shutdown and actual scoped R2 worker delivery are still required.

The execution service briefly disconnected during preparation. The prior retention check had passed, but an intended scope-file command failed before process creation. A provisional resume note was saved in configuration revision37. The input-audit agent recovered and launched its no-model verifier before learning of the pause; root then confirmed access and resumed automatically. That single verifier completed successfully. [The recovery receipt](INFRASTRUCTURE_RECOVERY.json) preserves this timing correction; final startup guidance supersedes the provisional note. No user reconnection or scientific rerun was needed.

All first22 evidence and tracked repository files remain unchanged. New evidence is retained locally and bound by the phase23 manifest. Saved startup configuration does not establish an external backup, publication or successful restoration in another environment.
