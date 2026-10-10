# Reuse of retained seed42 predictions

**PASS: the six phase16 seed42 prediction mosaics can be reused for the exact phase23 input contract.** All 150 independently reconstructed normalized tile hashes match the pre-forward captures. Each retained score mosaic is reproduced **bit-for-bit** from its 25 saved logits using the original FP32 sigmoid and Hann accumulation. No model was constructed or run.

All six inputs are already finite, little-endian FP32 `[17,384,384]` surfaces. They require planes0–16, `layer_start=0`, forward order, and no additional depth pooling, selection, reversal or padding. A row-major grid starts at each combination of x/y in `[0,64,128,192,256]`, producing25 `[17,128,128]` tiles per region. No supplied mask is passed to the adapter; every tile was nonempty and scored, and all147,456 pixels per context have retained computational coverage. Score evaluation still uses the smaller fixed primary mask below.

Normalization is the unchanged robust-MAD1/99 operation over the entire17×128×128 patch, with no label, geometry or nonzero mask. An independent NumPy implementation matched the retained captured input bytes for all150 tiles. Four synthetic cases first checked zeros, constants, a ramp and sparse/MAD-fallback behavior against the pinned pure normalizer. The original source and all focused vendor module hashes were verified without importing any model factory.

The arithmetic replay uses nonperiodic128-point FP32 Hann windows, outer product, maximum normalization and a .001 floor; sigmoid is applied to each tile's FP32 logits before FP32 probability-space blending. All six mosaics match exactly, including their complete evaluated masks. A separate float64 NumPy sigmoid/Hann calculation differs by at most **2.91559211607062e−7**, below the predeclared2e−6 numerical bound. These are raw FP32 sigmoid probabilities; neither uint8 TIFF quantization nor Nieuwlaar's `(value−64)/128` transform belongs in this comparison.

| Context | Parent / physical fragment | Primary positive | Primary negative | Primary total |
|---|---|---:|---:|---:|
| frag3-context1 | Paris1 / PHercParis1Fr34 | 9,434 | 82,951 | 92,385 |
| frag3-context2 | Paris1 / PHercParis1Fr34 | 5,771 | 83,490 | 89,261 |
| frag1-context1 | Paris2 / PHercParis2Fr47 | 19,400 | 69,159 | 88,559 |
| frag1-context2 | Paris2 / PHercParis2Fr47 | 12,633 | 70,681 | 83,314 |
| frag2-context1 | Paris2 / PHercParis2Fr143 | 19,049 | 61,660 | 80,709 |
| frag2-context2 | Paris2 / PHercParis2Fr143 | 20,248 | 66,318 | 86,566 |
| Total | Six reused development contexts | **86,535** | **434,259** | **520,794** |

The supplemental support check confirms each primary mask equals its retained known mask intersected with the central square `[32:352,32:352]`. Each known mask is a subset of the supplied surface mask. This is not simply `known & surface`: the32-pixel context border exclusion must remain. The inherited phase15 known-mask construction uses a symmetric33×33 native surface/label-homogeneity rule and samples at3×3 block centers; this audit does not reread the full native label canvases or revalidate annotation alignment.

The source surfaces were pooled previously from source planes7–57 in contiguous groups of3, with matching3×3 XY means: `float32(uint32_sum27 / float64(27*257))`. Their coarse depth centers are8,11,…,56, with source plane32 at coarse index8 and nominal9.72µm spacing. The retained surface file hashes are now rechecked; the raw source acquisition/pooling is not rerun.

The retained seed42 identity is HF `scrollprize/ink_9um` revision `7109667e2607db1b90c37c8b09cb876ea7fe7bb1`, `hybrid_3d2d-seed42/step-075000.pth`,138,360,039 bytes, SHA256 `e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab`; semantic state hash `21c186000b9b3093835e185342668acd0d86147a719c37513e937924b77ea228`. Both original phase16 workers record the same strictly loaded configuration, unchanged state and absent training/gradients. Artifact identity here is bound through their pinned receipts, the phase21 plan and root's prior retention verification; this narrow audit does not reopen or deserialize checkpoint bytes. `seed42-bound-config.json` retains their exact configuration for review.

The checked runtime remains Python3.12.14, NumPy2.5.3 and Torch2.12.0+cpu, with four CPU threads, one interop thread, deterministic algorithms and no autocast. The full audit passed519 assertions in1.762 seconds after process startup; the additional support pass checked all six exact mask formulas. It read only authorized existing development arrays and source/metadata, made no network request, and performed no checkpoint decoding, model calls or training.

`INPUT_CONTRACT.json` contains the canonical context order, source/output descriptors, complete150 normalized input hashes and bounding boxes, physical parents, support counts, seed42 identity and verified pins. `verification.json` binds the frozen verifier and results; `SUPPORT_CONTRACT.json` adds the exact primary-mask formula. Root decides reuse and any native execution separately.

All six contexts are **reused development evidence**. Computational reproducibility does not establish independent evaluation, renewed model accuracy, historical exposure independence or ink/label registration. The original phase16 “calibration” and “evaluation” labels are retained only as historical provenance.

A read-only source command initially failed to create because the exec-server transport disconnected. Tools subsequently recovered before a root stop instruction arrived, and the single verifier had already launched as shell session58421. No result was claimed while its completion was unobserved. After root authorized resumption, polling showed exit0/PASS; this was one verifier execution, with no model execution or scientific retry.
