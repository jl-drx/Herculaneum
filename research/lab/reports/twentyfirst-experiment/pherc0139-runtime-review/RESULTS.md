# Independent PHerc0139 public-geometry runtime review

**PASS on the first frozen verification attempt.** Both fixed PHerc0139 contexts satisfy the declared current-public mesh/stencil and native-frame interpolation-bound conditions at every reconstructed output. This is a computational geometry result, not authentication of historical private rendering or observed native CT/material support.

## Frozen execution identity

The reviewer contract and code were frozen before acquired-coordinate reads, with the scientific result confirmed absent. `CHECKER_FREEZE.json` SHA256 is `fc7d6c6ff7c038da25aef4e729fecaa1b98c5ae0dd395b771002cccd910ff1d7`; numerical contract SHA256 is `48f25b89c61753b1114a2fd75aa7b1d8c1bef3163743a25c6e0853dbced789be`.

The governing range/reconstruction proposal is SHA256 `4f180f4507e77c8460c5b3a57c0683e818c8b29957c1a6d75c70dc9d53f6623d`. The acquired result is `public-geometry-implementation-v2/run/result.json`, SHA256 `fcd516e1ea41a13cf5e6760fa898538a9aba2e25262ad8d4a1b5d342516f1a73`, bound to root GO SHA256 `fbbaa0337c1a6c06f3406d185ae7023ea5d6f4eac40176b6496e0f09f463491c` and producer v2 freeze SHA256 `b86b5faff357757e06f6fc9bc16ae29f6d60535885e9393419d6cb5a28c8d1aa`.

The producer's pre-acquisition durability correction was additive: v1 remains retained, while v2 binds durable reservation publication. Scientific geometry kernels and driver are unchanged. Reviewer production bindings were updated before their first freeze; synthetic-v1 and synthetic-v2 receipts are both retained. They contain the same 23 checks applied to their respective source generations, not 46 distinct vector-suite cases.

## What was independently checked

All six request extents were independently derived from pinned TIFF structural metadata: little-endian float32, uncompressed single strip beginning at byte 8, width 1444 and height 1404. Ordered identities, URLs, strong ETags, exact HTTP 206 ranges/totals, encoding, lengths, reservations, retained body hashes and result pins agree. Total retained range bodies are 2,739,744 bytes; total reservations are 2,739,750 bytes.

Only the six declared 80×80 channel windows were decoded: 80 row slices of 320 bytes each, at stride 5776. An independent struct-based selected-word decoder preserves all float32 bits, including signed zero and NaN representations. Both saved XYZ windows match those selected bytes exactly. The 153,600 useful coordinate bytes were interpreted; all 2,586,144 intervening bytes remained opaque. No additional coordinate extent was inferred from overlapping broad range spans.

A separately implemented 64-row NumPy reconstruction independently evaluated every fine-raster base, conservative four-vertex interpolation and five-vertex normal stencil, staged float32 normal construction and double normalization, and native-frame bounds. Separate horizontal/vertical logical reductions required all 16 fine constituents for each coarse output. No producer numerical implementation was imported.

| Independently reconstructed quantity | Rank01 | Rank02 |
| --- | ---: | ---: |
| Fine samples | 2,359,296 | 2,359,296 |
| Invalid position stencil or base | 0 | 0 |
| Invalid normal stencil or direction | 0 | 0 |
| Valid mesh outside declared native bounds | 0 | 0 |
| Fine samples satisfying mesh and bounds | 2,359,296 | 2,359,296 |
| Coarse mesh/stencil support | 147,456 / 147,456 | 147,456 / 147,456 |
| Coarse mesh plus declared native bounds | 147,456 / 147,456 | 147,456 / 147,456 |

All four complete boolean arrays, all disjoint rejection counts, subset and partition invariants, output shapes/dtypes and pins match exactly. The four all-true masks have raw-byte SHA256 `6765ef95316a0173ce56c8468d5d27d9a48b3fc1b5b3ac8f8c75d88c413eabd2` and identical `.npy` SHA256 `b72ed0084d6ba14f372c29db8e669805a285f71a0eb6aa5d3c3ccc5f62598260`.

A separately authored scalar struct-FP32 oracle also checked 169 fixed points per context, 338 in total. Finite bases and supported directions agree bit for bit with the vector reconstruction, and position/mesh/bounds decisions agree exactly. At the same fixed points, explicit enumeration of all 68 depths agrees with endpoint bounds. The complete-context depth coverage rests on the predeclared staged-arithmetic monotonicity proof; the reviewer did not execute all 68 depths at every fine point.

## Preparation and numerical acceptance

The vector/integration suite passed 23 checks, including complete generated flat/tilted planes, full-context bounds and degenerate-normal rejection, sentinel partitions, poisoned gaps, native boundary cases, all four dependency footprints and eight altered transport receipts. The separate scalar suite passed 23 checks, including 2,048 deterministic endpoint-versus-all-depth cases, reversed/tilted geometry, invalid axial neighbors, sentinels and escaped-window rejection. These are synthetic preparation evidence, not independent biological observations.

Acceptance was exact throughout: no floating tolerance, mask allowance, source repair or numerical-contract change followed actual output reads. No review STOP or scientific retry occurred. Source and input pins were rechecked after reconstruction.

The runtime receipt is `runtime-v1/verification.json`, SHA256 `95252dabe48ac22eb9f9aadcb90d33658cb7eea812ba180cef9e6129087d7466`. Reviewer elapsed time was approximately 0.790 seconds with 44,294,144 bytes peak RSS. The pinned producer result reports approximately 9.697 seconds and 33,640,448 bytes peak RSS. The reviewed producer directory retained 3,505,009 bytes at verification, below its 32 MiB cap. Producer resource checks were validated as retained evidence; the reviewer did not independently measure its transient peak RSS or test a real process crash.

## Limits and unchanged scientific scope

This pass establishes consistency of retained current-public geometry under the declared conservative staged arithmetic and original catalogue frame `[26511,26511,76953]`. It does not prove historical private compiler behavior, historical mask presence, historical rendering identity, physical registration correctness, observed native voxel availability, masked material/tissue support, papyrus, ink, labels or teacher ancestry.

The reviewer made zero network requests, native CT reads, label/S/V reads, model calls, checkpoint decodes or optimization steps. It did not change completed allocation-study masks, objectives, decisions or outputs. All-true computational masks do not authorize treating unknown pixels as background, expanding teacher support, rerunning scientific studies or renting a GPU. Root retains scientific interpretation and any future design decision.
