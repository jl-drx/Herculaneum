# Stage 2: strict native loading and synthetic CPU compatibility

Freeze this protocol, author code/plan and independent numerical checks before actual model construction or forwards. Stage1 must pass both artifact inspectors first. This is an implementation qualification with fabricated inputs, not an ink-accuracy experiment or a reproduction of historical GPU metrics. Seed42 and all first21 evidence remain unchanged.

## Implementations and state

Compare the exact active historical factory at Villa `3ea17f54a9b3d5fd1aaf73e1d2c8386dbaa9f30e` against the existing focused factory at `e0bbb8b40a2db58b1d71864f286eb85717e59e64`. The native source audit's nine-module closure is pinned by `HISTORICAL_FACTORY_CLOSURE.json`, SHA256 `d7250544daa1e74a6bef7cde882c6df415cd6714311793f12abdec9280a060c5`. Reused module bytes must match the historical Git blob identities. Preserve exact historical active module source. Namespace-only packages and explicit fail-closed inactive ResNet/Primus/DINO imports are allowed; assert zero constructor/function calls on these alternatives. Exact historical activations remain real code because active blocks compare class identities.

Construct each factory in its own fresh offline subprocess using the published native configuration and its documented inference normalization. Historical construction consumes the authored normalized dictionary; current construction uses its audited InkConfig parser. Capture the actual resolved construction properties. Use installed Python3.12.14, NumPy2.5.3, Torch2.12.0+cpu and safetensors0.8.0, FP32, evaluation/inference mode, four intra-op threads, one inter-op thread, and deterministic algorithms. No autocast, compilation, TTA, gradients or training. This runtime differs from the author's historical Torch2.4.1/CUDA12.1 runtime.

Load only the Stage1-verified safetensors file, with exact508 keys and `strict=True`. Check dtype/shape without conversion, all loaded bytes, 264 actual factory storage groups including244 duplicate-name pairs, and34,546,498 unique parameters. No partial load, key deletion, random missing state or new checkpoint conversion is allowed. Save per-key value hashes and alias partition before and after forwards and require unchanged state. Both factories must be entirely network-disabled before import/construction. Import-source hashes must match before and after.

## Fixed synthetic inputs and counts

For nonzero volumes use uint8 formula `1 + ((17*z + 3*y + 5*x + (y*x)%29 + (z*y)%23) % 251)` on a zero-based ZYX integer grid. Do not fit inputs to outputs. Use nine fixed cases:

| Case | Input ZYX | Requested bounds/direction | Expected final layer indices | Forwards per factory |
|---|---|---|---|---:|
| native-default |28×128×128|default, forward|6..22|1|
| native-window-0-20 |28×128×128|[0,20), forward|2..18|1|
| native-window-3-23 |28×128×128|[3,23), forward|5..21|1|
| native-window-5-25 |28×128×128|[5,25), forward|7..23|1|
| native-window-8-27 |28×128×128|[8,27), forward|9..25|1|
| aligned-default |21×128×128|default, forward|2..18|1|
| native-reverse |28×128×128|default, reverse after center crop|22..6|1|
| native-tiled |28×192×192|default, forward|6..22; XY starts0,64 per axis|4|
| zero-skip |28×128×128 all zero|default, forward|6..22; raw occupancy empty|0|

Exactly two factory constructions and two state loads, with11 full-model forwards per worker and22 total. Full-model component calls overlap; count the full-model entry explicitly. Process the two workers sequentially. No speculative extra run, warmup, replay, retry or intermediate checkpoint is authorized. Preparation fixtures may use generated arrays and toy models only, with no real factory/weight calls.

Use unmodified, explicitly selected safe historical inference functions/classes over in-memory arrays for its center selection, reader/dataset preprocessing, Hann weights, tiling, accumulation and output conversion; retain AST/source extraction provenance. Exclude the heavy CLI, unsafe torch.load, Zarr/network opening and writer side effects. The current comparator uses the retained baseline normalization, Hann and tiling behavior with the same selected indices and synthetic source tensors. Record the actual comparison boundary; it is not a full historical CLI or GPU replay.

## Measurements and frozen acceptance

Retain all selected indices, model input tensors, pre-sigmoid logits, probabilities, tile schedule and evaluated masks. Capture historical logits before its in-place sigmoid. Retain final raw blended FP32 probabilities separately from clipped-and-truncated uint8 `255*probability` and the author's subsequent clipped `(uint8-64)/128` decoding. Do not call the latter calibrated probability. Zero-skip must produce no model call and zero output, separately marked unevaluated.

Require exact indices, schedules, normalized tensor bytes, state bytes and evaluated masks. Both models must yield finite `[1,1,128,128]` FP32 logits and finite probabilities in[0,1]. Compare corresponding historical/current logits with elementwise absolute-plus-relative tolerance `1e-5 + 1e-5*abs(historical)`, and raw probabilities/blends with `2e-6 + 2e-6*abs(historical)`. Report maximum absolute and relative differences and exact-equality counts even when tolerance passes. Tolerances are fixed here, before any native model output. A failure stops qualification; do not change source, dtype, input selection or tolerance in response.

An independent checker reconstructs sigmoid, weighted accumulation, normalization, quantization and author decoding from saved evidence using frozen arithmetic tolerances. It verifies loaded per-key byte hashes against independently read safetensors values, exact model-call counts, source and state invariants, and every comparison criterion. It does not rerun the models. For each implementation, saved encoded outputs must exactly match its own recorded conversion arithmetic. Cross-factory uint8/decoded mismatch counts are diagnostics because a small accepted floating difference can cross a quantization boundary; report them without changing raw-score acceptance.

Meaningful toy tests must cover wrong center/odd-even bounds, reversed selected depth, raw-zero skip, constant normalization, padding selection and normalization ownership, clipped quantization endpoints, overlap/Hann normalization, mismatched aliases/keys/state hashes and shape/count rejection. No new factual training or performance claims follow from these fixtures.

## Resources and interpretation

Each worker: at most180seconds wall, 360CPU seconds, 4GiB measured RSS and8GiB address space. Controller:400seconds external wall. New Stage2 outputs at most128MiB and entirephase22 at most768MiB. Preserve failures; no automatic retry. Full native file already exists from Stage1, so this stage makes zero acquisition requests.

PASS establishes strict loading and the named CPU synthetic compatibility coverage only. It does not establish historical CUDA equality, unknown preprocessing branches, out-of-distribution accuracy, physical ink, teacher independence, independent confirmation or improved Plinius. The benchmark provenance audit remains a separate prerequisite for any candidate scoring. No paid launch follows automatically.
