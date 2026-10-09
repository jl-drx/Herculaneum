# Plinius phase 22: native checkpoint qualified; independent evaluation still unresolved

The published dense-native checkpoint is now qualified for the tested CPU inference path. Both exact historical and retained current factories load it strictly, preserve every parameter byte, and produce **bitwise-identical outputs across all nine frozen synthetic cases**. This removes a concrete implementation uncertainty before a real-data comparison. It does not establish improved ink accuracy or historical CUDA equivalence. Seed42 remains our adopted baseline; no new scientific images were scored, no training occurred and GPU spend remains £0.

The evaluation audit did not establish an independent accuracy reserve. PHerc0841 adds a locally unused labeled parent, but its model-reviewed/reconstructed annotations and incomplete public mapping/provenance package prevent treating it as authenticated optical truth. The next useful experiment is a single, fixed native-reference comparison on our existing development inputs, with its repeated-use limitations stated explicitly; further adapter tuning and GPU scaling are not justified yet.

## Native artifact and inference result

The exact artifact is `Nieuwlaar/ink9um-dense-native`, revision `b894a0493ab2291cc8dc3247020c0d7cb0967a58`, file `weights/dense_native_016000.safetensors`. All **272,767,904 bytes** match SHA256 `d3d95dc41dcb24ed76b84b5e12ca5dae25d7fb34edf1dfbb93e0186f91e6fc6d`. Independent raw-format and safetensors readers verified 508 named FP32 arrays, all 68,175,426 named elements finite, and byte equality of every one of 244 expected duplicate-name pairs. Strict construction subsequently confirmed 264 actual factory groups and 34,546,498 unique parameters. Duplicate serialized names are not a claim that the file itself shares storage.

The acquisition used one logical full-file request with one counted redirect: two HTTP attempts, 272,769,016 observed response-body bytes and 131.263 seconds under the external controller. No acquisition retry or checkpoint conversion occurred. The advertised transport ETag was checked separately from the full-file SHA256. Both inspectors passed before any actual model construction.

The source audit recovered the native author's historical active factory at Villa `3ea17f54a9b3d5fd1aaf73e1d2c8386dbaa9f30e`. Nine active module identities are pinned; reused files match exact historical Git blobs. Twenty selected historical inference definitions run unchanged over synthetic in-memory arrays. Namespace shims and fail-closed inactive architecture imports do not substitute for active model code. The comparison excludes the historical CLI, unsafe checkpoint deserialization, network storage and writer side effects. It uses the retained current focused factory at `e0bbb8b40a2db58b1d71864f286eb85717e59e64` as the second implementation.

The frozen test covers native default depth, four explicit windows, a 21-plane aligned input, reversed depth, overlapping tiles and empty-input skipping. It performed exactly two constructions, two strict state loads and 22 model forwards in separate sequential offline workers. Each worker verified all loaded state bytes and aliases before and after inference; no state changed, gradients formed or forbidden network/unsafe-load call occurred.

| Independent comparison | Values compared | Result |
|---|---:|---|
| Pre-sigmoid tile logits | 180,224 | Bitwise identical; maximum absolute difference 0 |
| Tile sigmoid probabilities | 180,224 | Bitwise identical; maximum absolute difference 0 |
| Raw blended output values, including empty case | 167,936 | Bitwise identical; maximum absolute difference 0 |
| Selected indices, normalized tensors, tile schedules, evaluated masks | All nine cases | Exact |
| Historical uint8 and subsequent clipped-author outputs | All nine cases | Zero cross-factory mismatches |

The independent checker reconstructed normalization, selection, sigmoid, Hann blending and output conversion without replaying either model. Independent arithmetic differs by at most 5.96e-8 for sigmoid and 1.79e-7 for Hann weights, within the frozen tolerances; reconstructed accumulation sums, weights and raw scores are exact. The two model workers completed in **8.465 seconds** externally, with maximum self-reported peak RSS below 827 MB. Runtime was Python3.12.14 / NumPy2.5.3 / Torch2.12.0+cpu / safetensors0.8.0, FP32 with four intra-op threads. The author's Torch2.4.1/CUDA12.1 runtime was not reproduced.

Preparation passed 94 author toy checks, 32 independent evidence fixtures and 23 independent source/preprocessing comparisons. These are distinct from the actual native forwards and do not count as scientific samples. Full evidence is in [the artifact report](checkpoint-qualification/RESULTS.md), [independent artifact review](checkpoint-review/RESULTS.md), [input contract](native-input-contract/RESULTS.md), [model execution report](model-parity/RESULTS.md) and [independent parity review](parity-review/RESULTS.md).

## Input semantics that matter for future experiments

The historical default selects **planes 6–22 from a 28-plane volume**, and planes 2–18 from a 21-plane volume. Explicit source bounds 0:20, 3:23, 5:25 and 8:27 select 2:19, 5:22, 7:24 and 9:26 respectively; depth reversal follows center selection. These are array indices, not proof of physical ink depth or annotation registration. Do not center-crop an already selected 17-plane tensor a second time.

The historical saved TIFF truncates `255 * blended_sigmoid` to uint8. The author's later score decoding clips `(uint8 - 64) / 128` to [0,1]. That quantization/clipping changes ties and score meaning. Keep raw FP32 results distinct from this published representation, and do not describe the decoded values as calibrated probabilities. Training and inference also differ in how padding participates in normalization; the actual native test exercised full-size inputs, with padding behavior covered only by toy/source checks. Compatibility on these cases is not exhaustive branch coverage.

## Evaluation provenance

The retained official catalog contains 33 explicit modern ink-label records across seven parent scrolls. PHerc0139, PHerc0814, PHerc1667 and PHercParis4 occur in the original models' declared training configurations. PHerc0009B, PHerc0500P2 and PHerc0841 require more careful classification than a new crop or absent config entry can supply.

PHerc0009B and PHerc0500P2 are absent from the inspected original/native root lists but explicitly present in the paper R3D50 model's training evidence. That model is distinct from the canonical production R152 teacher. No retained artifact proves a teacher ancestry edge from the paper model to seed42's original labels; neither independence nor that specific contamination should be invented.

PHerc0841 is absent from the inspected original/native roots and the inspected paper pipeline. Wider ancestry remains unknown. All three published label records are transferred, have unknown original annotation surface, and were reviewed against predictions. One explicitly reports aggressive iterative completion and reconstructed letter forms. The complete-prefix metadata checks found **39 objects in each of three packages**, limited to inklabels, supervision and validation assets. No package advertises a mapping-validity object or the `provenance.json` linked from all three ink-root metadata files. This is a demonstrated gap in those exact complete inventories, not proof that no provenance exists elsewhere.

The initial metadata request failed before an HTTP response; its reader bypassed the configured proxy, and the recorded URLError did not preserve its underlying cause. The root preserved that attempt, diagnosed the proxy bypass from code and variable-presence checks, froze an additive corrected transport, and explicitly authorized one revised run. All six revised requests returned HTTP200, totaling 59,771 observed response-body bytes. No array payloads, predictions or labels were fetched. There was one explicit revised acquisition following the setup correction, with no automatic retry loop.

Frag6 remains a potential optical-reference route, but historical rendering, annotation/registration provenance and negative-label support are unresolved. Earlier sparse geometry success does not open Stage C. Frag5 shares the known-trained PHerc1667 parent. All six Paris contexts are reused development; Paris1 and Paris2 are two physical parents, not six independent confirmations. See [the eligibility inventory](benchmark-inventory/RESULTS.md) and [corrected metadata findings](benchmark-metadata/v2/RESULTS.md).

## Decision and retained limitations

Proceed next with the [fixed-reference study design](NEXT_STUDY_DRAFT.md): qualify the exact retained input contract, compare only seed42 and this native checkpoint, avoid threshold/depth/ensemble searches, and report descriptive performance and disagreements on existing development contexts. Keep locally unscored candidate parents reserved. A future supplied-reference agreement study and an optically anchored accuracy study require different evidence and should be labeled accordingly. There is no authenticated public endpoint yet for several missing historical teacher/render/label manifests; repeated unchanged metadata queries cannot solve that gap.

This phase made no transcription, translation, model promotion, prize submission, external publication, credential change or upload. Native code/model licensing evidence is retained, but this is not a current prize-eligibility determination. The £10 total first paid pilot cap persists across retries and overheads. Actual scoped R2 secret delivery to a Runpod worker remains unverified; frozen workload, pinned CUDA, current pricing and independent shutdown are still required before paid execution.

All first21 sealed evidence and tracked repository files are preserved. Two preparation fixture failures, a root prelaunch status-enum correction and an independent receipt path-normalization correction are retained; none reran actual native values or model forwards. The metadata transport failure and explicit corrected run are also retained. Local retention hashes and saved startup instructions establish current evidence/configuration state, not external backup, environment publication or proven fresh-task restoration.
