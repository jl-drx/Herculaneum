# Independent PHerc0800 geometry-support audit

## Decision and useful new finding

The frozen conservative admission decision is reproduced: **none of the 40 retained components contains a 128×128 core under the Chebyshev-radius-128 rule**. All 40 component rankings, conservative areas, largest-square sizes and coordinates match the retained reference exactly.

A separate, explicitly conditional diagnostic changes the field-review outlook. The one permitted native grid (128×128 patches, stride 64, origin (0,0), including the final length−128 start) retains a **192×192 core in segment 20251028225813, component 1, at (x=1024,y=256)**. This is the first component in the frozen order. Its conservative maximum is only 69×69 at (1093,318).

This is a geometry/exposure review candidate, **not evidence of ink, absence of ink, model accuracy, or admission of the full four-model/plain+padded recipe**. In particular, Hecate rescaling and padding require exact contributing-footprint qualification before that recipe can be admitted. No alternate grid origins, extra depths, target intensities, models, GPU work, paid compute, new workflow runs, repository edits or original-task messages were used.

## Independence, provenance and reproducibility

Source: [GitHub Actions run 38065705596](https://github.com/jl-drx/Herculaneum/actions/runs/38065705596), artifact 11674952894, 693,653 bytes. Downloaded ZIP SHA256: `8d73a97c6fba42436d1bdb0397cbb00ba59822e437bf2a6b4998d66fafe523ea`.

The ZIP was checked and extracted with traversal, duplicate-path and symlink rejection and an uncompressed-size bound. SHA256SUMS, all 57 packet-manifest records, and all input-manifest records were verified. All 43 actual source input files match frozen GO source pins. Five renderer metadata records match the pinned commit and unrotated, uncropped, no-affine source-export conventions. The historical phase47 cumulative seal and immutable workspace fallback manifest are only identified by hash in this packet, not included; their full contents could not be independently reopened.

START_HERE, the protocol, GO and data schemas were read first. `audit.py` and its checksum were frozen before opening reference implementation or expected outputs. The initial five geometry masks were saved independently. A deterministic component-sort bug then prevented final JSON serialization; `summarize_saved.py` recovered the summaries from those unchanged masks. The correction and a strict-threshold KD-tree query-bound fix were documented and frozen in `audit_locked.py` before reference inspection. The independent result hash was frozen before opening expected results.

The final full replay used Python 3.12.14 and the exact hash-locked NumPy 2.5.3, SciPy 1.18.1 and tifffile 2026.9.20. Every saved independent mask, receipt, mesh count and component result is bit-identical to the first environment. The locked replay took 269.1 seconds and peaked at **1,650,176 KiB (1.574 GiB)** RSS, well below 7 GiB. Four KD-tree workers and single-threaded BLAS/OpenMP were used. No swap was available or used. A separate post-reference arithmetic reconciliation peaked at 1,449,480 KiB. Dependencies were isolated under the audit directory.

## Numerical checks and disagreements

- Historical old-centre union before deduplication: **3,260,832**, exactly as frozen. Extent contributions: 497,600 (29010146), 2,196,160 (28222030), and 567,072 (28220955). All were regenerated from coordinate TIFFs using float64 interpolation followed by one float32 cast.
- All 16 archived point receipts pass: maximum base-component error **0.001953125 voxel** (limit 0.002), maximum normal-component error **5.96046448e−8** (limit 1e−6). Full historical-versus-candidate base-map error is 0.001953125 on each larger mesh (limit 0.01).
- The independent float32 expression and the precisely archived expression are **not point-mask-identical**. There are **nine changed pixels**: 0, 3, 2, 1, 3 by segment order below. Their net independent-minus-archived point-count changes are **0, +1, 0, −1, +1**. Six changes cross the 51.1 nearest-centre threshold; three cross an integer CT-box envelope boundary. Exact coordinates, positions, distances, envelopes and old-box indices are in `discrepancy-fixtures.json`.
- A separately labeled, post-reference reconstruction of the reviewed archived operation order matches **every retained stage count exactly**. It is convention reconciliation, not independent evidence. It does not import or execute the packet's scientific scripts.
- Crucially, **both complete context masks are bit-identical between arithmetic conventions on every mesh**. Every conservative component result and the conditional fixed-grid result are therefore unchanged. Both expressions support every pixel of the preferred candidate's covering-patch union.
- Independent tests include 90 randomized exhaustive largest-square/tie checks, 500 direct all-covering-patch pixel checks including irregular final starts, and 10,000 randomized normals × all 66 layers confirming endpoint integer envelopes enclose float32 trilinear corners. The conservative mask was explicitly checked to be a subset of the exact fixed-grid mask on all five canvases.

## Counts and complete component coverage

Context areas below are whole-canvas areas before phase47 component restriction; square maxima are **within retained components**. Consequently, the 20955 canvas has nonzero conservative context but no conservative pixels assigned to the retained components. `component_metrics.csv` records all 40 components, including every failure, in frozen selection order.

| Segment | Point support, independent / archived | Context area, conservative / fixed grid | Largest restricted square side, conservative / fixed grid |
|---|---:|---:|---:|
| 20251028213516 | 251,463 / 251,463 | 0 / 0 | 0 / 0 |
| 20251028220042 | 184,697 / 184,696 | 0 / 0 | 0 / 0 |
| 20251028220955 | 690,233 / 690,233 | 11,754 / 53,248 | 0 / 38 |
| 20251028222030 | 710,587 / 710,588 | 8,160 / 98,304 | 35 / 128 |
| 20251028225813 | 369,450 / 369,449 | 11,442 / 53,248 | 69 / 192 |

The three conditional 128×128 alternatives are segment 20251028222030 components 2, 1 and 4, respectively at (256,1216), (1088,256) and (1280,2176). All other components have a fixed-grid maximum below 128. No candidate was selected by trying additional origins or depths. `stage_metrics.csv` separates losses from normal validity, native bounds, old-surface proximity, old-CT boxes and context. Sequential loss columns are order-dependent; old-box intersections measured on all valid geometry are also supplied and must not be naively added to overlapping exclusion counts.

## Preserved preferred candidate and margins

Coordinates are zero-based native source pixels, with half-open rectangles:

- Scored core: **[1024,1216) × [256,448)**, 192×192.
- Exactly 16 contributing 128×128 patches: x starts **960, 1024, 1088, 1152**, crossed with y starts **192, 256, 320, 384**. The complete enumerated list is in `candidate.json` and the independent result JSON.
- Full patch union: **[960,1280) × [192,512)**, 320×320 = **102,400 pixels**. All source pixels pass unchanged normal, depth, native-boundary, prior-surface and seven-box exclusions under both expressions. Only the scored core, as prescribed, must belong to phase47 component 1; all contributing context is independently checked at full resolution.
- Minimum nearest-old-centre distance across the whole union: **53.703812490 voxels** with archived arithmetic and **53.704709077** independently. The smaller margin above 51.1 is **2.603812490 voxels**.
- Minimum native-volume boundary index gap for the inclusive integer corner envelopes: **3,036 voxels**, in both expressions.
- Minimum separating-axis integer index gap to the seven old boxes, in GO order: **1171, 454, 711, 711, 711, 327, 210**. The smallest is **210** in both expressions. This is an AABB separation measure, not a remeasurement of physical-sheet distance or the exact CT sampling kernel.

`SUPPORT_COMPARISON.png` shows all five canvases and the preferred candidate. Black is the conditional core, purple dashed is its source-patch union, and the white square is the 69px conservative maximum. Orange/green displayed areas are whole-canvas context masks; component-restricted results are in the CSV.

## Assumptions, scientific limits and consequence

The 51.1 bound remains sufficient for the specified ±32.5 proposed and ±15 previous slabs. A units nuance in the prose is worth recording: if 0.01 base error is per coordinate, the two-position Euclidean allowance is 0.02√3 rather than 0.02. The corresponding sufficient bound is **50.998742631**, still below 51.1 by **0.101257369**. This does not change admission. Failure of the bound is unresolved proximity, not proof of overlapping voxel support or the same physical sheet.

The integer slab AABB, whole previously available chunks, radius-128 all-origin context and nominal level3 scaling are conservative. Level3's true multiscale reconstruction kernel was not rederived. Float rounding can alter isolated threshold/box-boundary pixels despite bounded base error; that is why discrepancies are retained rather than suppressed. The chosen patch union has much larger measured margins.

The result qualifies geometry only relative to the three recorded surface extents and seven recorded boxes. It cannot establish that all CT objects exist, renderer fidelity at execution, unique physical-sheet identity, independent ink ground truth, complete training exclusion, or absence of undocumented/public human exposure. A fresh coordinate field on the already-studied PHerc0800 specimen is not a reset of specimen independence or a pristine held-out benchmark.

**Practical consequence:** preserve this exact field and patch list for a narrowly defined renderer/model-footprint review. The frozen conservative study still ends in no admission. The single-grid diagnostic prevents that conservative failure from being misread as “no useful fresh field exists,” but does not authorize silently weakening the original gate or starting target inference. No inference was performed.

## Reproduce and inspect

The ZIP includes the sealed packet, independent code and freeze checksums, corrected locked implementation, explicit post-reference reconciliation, environment record, complete metrics, discrepancy fixtures, compressed masks and figure. Dependencies are not bundled. In a fresh Python 3.12 environment, install `packet/requirements.lock` with `pip --require-hashes`, then run `audit_locked.py`. `reconcile_archived_convention.py` is explicitly post-reference; its comparison expects the retained independent `output/` masks. `audit.py` is preserved verbatim as the originally frozen implementation, including the documented final-sort bug; use the corrected locked runner for a full replay. The packet's original scientific scripts are evidence, not portable runners.

All output and source files are enumerated in the return ZIP's SHA256SUMS. The ZIP itself has a separately reported SHA256.
