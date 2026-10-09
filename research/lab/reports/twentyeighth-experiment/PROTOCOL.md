# Phase28: one-pass chunk accumulation and a focused research design

User agreed to finish data loading, establish a reliable reference experiment, focus on surface-depth uncertainty, then assess the £10 GPU pilot. This phase executes the remaining CPU scheduler qualification and prepares a concrete research design. It does not acquire reserved candidate pixels, perform model inference/training, send outreach or rent resources. Existing checkout and all27sealed phases stay unchanged. Prior retention anchor257fb82f9ad8de9ca660535659f71d9e2c1d87f5ee46c1468678923a30a73f18 must pass first.

## Prospective mechanics criteria

Reuse the frozen phase26 interpolation primitives and phase27 strict local reader. Build a new complete positive-weight contribution plan, in canonical ZYX chunk order, then process at most16native chunks /64MiB physical source work per batch. Each unique chunk is called once; only one returned source chunk remains resident in the scheduler at a time. Reader codec output/crop copies and process RSS are counted separately. Planning metadata/output accumulators are not source chunks. Native chunks remain128cubed little-endian uint16 with the published Blosc Zstd declaration.

Bounds: at most10000points,128total unique chunks,512MiB total planned physical source work,16MiB NumPy plan buffers,64MiB physical source work and128logical chunks per batch. Smaller batch caps in tests are permitted. No plan limit is a full-array or full-scroll scalability claim. Retained geometry requires2187slots and70chunks; expected schedule16+16+16+16+6. The whole-array shape may be huge, but synthetic writes/reads MUST be limited to these70chunks. Never allocate or read a full native-shaped array.

For each point retain expected positive-corner count, processed count, geometric validity, missing-source flag and nonfinite-source flag. Publish finite values only when all expected contributions are processed and every contributing source is known/finite; valid zero remains zero. Invalid geometry, missing source and nonfinite source are separate statuses. Execution interruption/corruption is terminal: no partial result returned, no accumulator checkpoint/resume implemented. A fresh full invocation is required after diagnosis. Cache atomicity does not imply accumulator recovery.

Check plan identity before loader calls and reject altered schedule/arrays/limits. Copy caller inputs and hold plan arrays read-only. Loaded edge shape and dtype must match, and a physical chunk must fit one batch. Bound before any source access. Use float64 weights/accumulators and original output slot order. Prospectively require exact validity/status/coverage equality and absolute numerical error<=1e-10 (rtol0) against existing direct sampler/scalar oracles and analytic phantoms. No outcome-based tolerance changes.

Test known linear and nontrivial multilinear phantoms, random off-grid points with fixed seeds, boundaries/corners/zero-weight neighbors, invalid geometry, missing/nonfinite data, duplicates, scalar axes negative control, dtype/endian/shape mismatches, budgets, changed plan, interrupted execution and no live source-buffer accumulation. The unchanged phase27 reader already has codec/cache tests; add a composed corruption-stop test rather than duplicate its entire suite.

## Native-layout synthetic comparison

Reuse ALL729geometry records (558mapped/171invalid), identical depth-major2187slots and hypothetical[-3.24,0,+3.24]um probes at3.24um spacing. This is still an unverified candidate-source-frame interpretation, not candidate sampling. Compare against the exact13contiguous ranges retained in phase27. Both paths use the same generated70sparse encoded objects, no cross-batch cache, and phase27 reader. Encode values z+2y+3x (uint16; volume-wide range fits) at source voxel centers. Reference Zarr readback must exactly match each written chunk. No native pixels enter this fixture.

Required result: identical point validity, every expected contribution exactly accounted for, tolerance passed, baseline200source reads/decodes versus optimized70 and5batches. Report measured encoded bytes, elapsed time and process RSS, but do not infer real scan compression, network speedup or model throughput. Save outputs, source identities and all attempt results. Root conducts a scalar/saved-result second pass; no independent agent is claimed.

No network, credentials, GPU/model calls. Per command timeout120seconds; phase artifacts32MiB cap before final manifest and32MiB including it. Freeze sources before the qualification run. Development attempts and any corrected failed frozen attempt remain distinguishable. No modifications after final sealing.

## Scientific design deliverable

Prepare a narrow H2 surface-position uncertainty experiment using the unchanged forward seed42 baseline. Do not impose depth reversal invariance, optimize directions per region, promote the native checkpoint, or call familiar development contexts held-out. Separate deliberately perturbed surface-position robustness from authentic ink accuracy. Freeze a small plain-language design plus machine-readable pending-reference contract with conditions for scoring and GPU use. Historical Frag6 render/optical/exposure gaps remain unresolved; a new documented rendering/optical reference is an alternative to historical reconstruction. Include concrete next actions and stop conditions without repeated unchanged provenance searches.

£0spent/£10TOTALfirstGPUcap; useful frozen job,pinnedCUDA,currentall-inpricing,independentshutdown and actual scopedR2delivery remain launch prerequisites. Maintainer submission/reply is unconfirmed.
