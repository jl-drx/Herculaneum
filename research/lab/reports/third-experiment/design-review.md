# Review of the larger PHerc0800 experiment design

Prepared 8 October 2026 from the retained live inventory, existing surface-coordinate methods, pinned detector contract and local rendering-consistency evidence. This review does not score new pixels. The defensible next experiment is a geometry-selected larger native surface context, with fixed centered inference and predefined depth controls. A different segment identifier alone cannot establish an independent physical region.

## Existing geometry already shows a shared CT neighborhood

The old small segment is `20251029010146`, mapped to eligible native volume `20250521135224` at 8.640 µm. Its mesh XYZ bounding box is:

```
minimum: [3089.9785, 5907.0840, 11617.8838]
maximum: [3703.0190, 6614.2456, 12497.9766]
```

An independent check of the retained `live-target-inventory.json` establishes that **each of the five larger segments' mesh XYZ bounding boxes completely contains this old box**:

| Larger segment | Native surface Y×X | Old mesh AABB contained |
| --- | ---: | --- |
| `20251028213516` | 2160×2240 | Yes |
| `20251028220042` | 2040×2000 | Yes |
| `20251028220955` | 2100×2100 | Yes |
| `20251028222030` | 2580×2580 | Yes |
| `20251028225813` | 2300×2260 | Yes |

This is evidence of a common CT neighborhood, not proof that the surfaces coincide. A bounding box contains empty space, folded material and potentially several adjacent papyrus wraps. Frame the work as a **larger spatial context or geometry extension** until coordinate-footprint comparisons support a more specific statement. Neither nonoverlapping surface-canvas rectangles nor different segment IDs imply disjoint physical papyrus.

## Measured overlap for the selected geometry-only crop

The frozen first crop is segment **20251028222030**, absolute native source XY **[512,512,1536,1536]**, within the 31×2580×2580 source. Its 1536-square canvas has **2,196,160 valid mapped pixels (93.0854%)**, 468/529 fully geometry-valid scheduled model tiles, and a nominal side of 13.27104 mm. The selection record retains 220 ranked candidates across both sizes and explicitly recommends the best 1536-square candidate first; this is rank 1 among 1536 candidates and rank 30 in the combined-size ranking. This recommendation preceded source-pixel reading or detector inference.

An independent, read-only `scipy.spatial.cKDTree` calculation used **every valid dense mapped pixel** in this selected crop and every valid dense pixel in the old small surface. No coordinate subsampling, image intensities, model outputs, fitted alignment or extra tolerance was used. Queries ran in 131,072-point batches with four workers. Both directions gave zero matches at the fixed overlap thresholds:

| Nearest-coordinate comparison | Dense query points | Targets | Distance ≤1 voxel (8.64 µm) | Distance ≤3 voxels (25.92 µm) | Median distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected crop → old tiny surface | 2,196,160 | 497,600 | 0 (0%) | 0 (0%) | 175.1292 voxels / 1513.1164 µm |
| Old tiny surface → selected crop | 497,600 | 2,196,160 | 0 (0%) | 0 (0%) | 31.9847 voxels / 276.3480 µm |

The global closest pair is **7.4620923 native voxels (64.4725 µm)** apart. Selected→old distance percentiles 1/5/50/95/99 are 15.6465/23.3282/175.1292/467.7304/576.6920 voxels; old→selected values are 11.9243/16.8569/31.9847/48.1490/52.0270 voxels. The asymmetry reflects the larger mapped context versus the small old footprint.

For normal comparisons, local tangent normals were derived as normalized `cross(P[x+1]−P[x−1], P[y+1]−P[y−1])`, requiring the point and all four neighbors to be geometry-valid. This is a dense-map tangent diagnostic, not the historical renderer's nearest cached vertex normal or its flipped depth normal. There are **no point pairs in the predefined 1- or 3-voxel sets**, so matched-normal angle and signed separation summaries for those sets are unavailable. No normal-based layer-identity claim follows.

These results support **no near-coincident base-map footprint at the stated tolerances**. They do not prove that the new papyrus surface is a different biological layer or that its model inputs are independent: the closest separation lies within the possible combined ±8-voxel primary depth spans, and trilinear CT neighborhoods can be shared. The correct description remains a broader PHerc0800 context with measured base-coordinate separation.

All four mask/map hashes were checked against their source summaries before computation and independently rechecked after computation; the files remained unchanged:

- Old full geometry mask: `a5ec3b003135871f517528f789cf62f112a79b4ac99f5e5324bb4162faea41f8`.
- Old full dense XYZ map: `639805f7062e5ce2351efaf77d43e226a0dca3151af8428c21fe3e70bff51c90`.
- Selected crop geometry mask: `a57b5566feb21f2052274e1dea8d474b1f35e59b40d8557480364a4b933cbdef`.
- Selected crop dense XYZ map: `63b27f31c7d014c00981fcc32f8d9ee8df86f276e0125f9f157583102769bb8b`.

Geometry evidence is retained in [the frozen candidate selection](geometry-selection.json) and the new mesh/source summaries under `data/experiments/pherc0800-larger-geometry/20251028222030/`. The dense overlap calculation took about 32 seconds after map/normal preparation, without any new network request or source-image read.

## Geometry-only physical overlap diagnostics

Use the transformed TIFXYZ meshes explicitly mapped to the **same native volume**. Preserve input hashes, CT coordinate order XYZ, native voxel units, mesh scale, canvas origins and crop XY coordinates. Do not compare untransformed acquisition coordinates to the native-volume maps. Retain the old full mask and dense map unchanged:

- `data/experiments/pherc0800-surface-geometry/full-surface-valid-mask.npy`;
- `data/experiments/pherc0800-surface-geometry/full-surface-ct-coordinates.xyz.npy`.

The old mask has 497,600 valid mapped pixels. It covers the previously scored small field, beyond the originally displayed pilot/hotspot crops, so use the **whole old valid footprint** as the overlap reference. Otherwise a new crop can be called uninspected simply because it avoids the old display window while still revisiting the scored field.

Perform these diagnostics before viewing new source intensities or predictions:

1. Use native XYZ bounding boxes as a broad-phase check. Disjoint boxes exclude coincidence of their base-surface points; overlapping boxes require the more precise test. Include a stated margin when assessing through-depth sampling.
2. Compare valid dense **base-surface coordinates** with bidirectional nearest-point distances, for example a SciPy `cKDTree`. Report new→old and old→new separately: a large extension can contain nearly all of the old footprint while only a small fraction of its own area is near it. Use all valid coordinates where practical, query in bounded batches and preserve nearest-match indices. If subsampling is needed, fix stride/origin before comparison and disclose that the result is a sampling diagnostic rather than an exact area fraction.
3. Use the root task's fixed primary overlap distances **1 and 3 native voxels**, corresponding to **8.64 and 25.92 µm**, and report cumulative matched fractions at each, together with count, minimum, median and distance percentiles. State the valid-point denominator and whether counts are dense mapped pixels or sampled points. Additional distance bins can be descriptive but must be declared before their results are interpreted. Do not turn a favorable threshold into an optimized overlap test after seeing results.
4. Retain the distance map over the new crop, and show the old footprint/nearest matches alongside it. Near-coincident points should have coherent nearby surface correspondences rather than scattered matches across unrelated folds. Inspect local tangent/normal alignment, signed normal separation and continuity of the matched old coordinates. A nearly parallel nearby wrap may still be a distinct physical surface. Agreement of normal directions alone is insufficient; absolute-dot alignment also suppresses a sign distinction that should be recorded explicitly.
5. Separate **near-coincident base geometry**, **nearby surfaces**, and **geometric extension beyond the old footprint** in the report. A small Euclidean distance is not an archaeological identity certificate. A large fraction with close, smooth, aligned correspondences supports geometric overlap; it does not prove a biological-layer topology or an independent label truth.

Base-coordinate separation also differs from CT-input independence. The primary 17-layer inputs sample offsets −8 through +8 voxels around each base point. Two parallel base surfaces can have overlapping input slabs despite distinct base coordinates. The retained full 31-layer stacks span −15 through +15. Compare the actual documented normal-offset sample coordinates when making a scan-voxel reuse claim; include trilinear interpolation support. A generous base-distance warning band of roughly 16 voxels for two primary windows, or 30 for two full stacks, only flags possible shared sampling; it does not prove or rule out overlap for arbitrary normal directions/folds. Even geometric nonoverlap within one scan does not yield statistically independent validation observations.

Report the selected crop's overlap diagnostics before claiming a previously unexamined physical area. If the selection is mainly an extension of an existing patch, keep that accurate description and report the portion outside the old near-coordinate footprint separately. An overlap exclusion mask may be useful as a secondary geometric summary, but must be frozen from geometry and must not manufacture an accuracy test without target labels.

## Absolute geometry, source and boundary masks

Reconstruct the base map with the documented renderer convention:

```
grid_x = (native_source_x + 0.5) * scale_x
grid_y = (native_source_y + 0.5) * scale_y
```

For these meshes the recorded scale is X/Y `[0.05,0.05]`, corresponding to 20 render pixels per stored grid interval. Add the crop's **absolute source XY origin** before evaluating this mapping. Confirm metadata scale/order and historical rendering overrides rather than assuming every new export matches the small segment. Require a complete in-grid bilinear cell and four finite, nonsentinel XYZ corners. The old source's positive-coordinate criterion matches its finite/non-minus-one validity rule because all genuine coordinates are away from volume origin; do not generalize a positivity rule to unrelated scans without checking their sentinel convention. Check mapped points against native CT bounds and source vertex bounds.

Keep separate masks with explicit meanings:

- **Geometry-valid:** complete trusted bilinear mapping, independently of intensities or model scores.
- **Source availability:** observed stored chunks versus explicitly reconstructed documented Zarr fill. A network failure is not missing-data fill. Missing geometry and zero source intensity are different conditions.
- **Selected-layer observed support:** any nonzero raw intensity among the fixed input layers 7:24. This is a conservative data-support flag, not papyrus/ink segmentation or proof that zero means no ink.
- **Evaluated:** a pixel received an actual model contribution for each seed. An empty-tile output zero with evaluated=false is not a learned no-ink response.
- **Boundary review support:** declared distances from crop edges and geometry/source-invalid regions, derived without predictions.

Keep the primary numerical mask as geometry-valid AND the documented source support AND both evaluated masks AND the frozen boundary policy. Retain counts for every exclusion and all component masks. Invalid scores remain NaN. Post hoc masks based on high scores or visually attractive texture do not belong in the primary mask.

Use an **absolute 64-pixel crop-edge margin** as a simple primary display/summary policy, applied equally to center and depth controls. Do not replace it with a changing percentage of canvas size. For a 1536 crop, the rectangular interior is 1408×1408; for 2048 it is 1920×1920. Geometry holes and source gaps require their own support distances; a crop-edge margin alone does not protect them.

A 64-pixel distance from invalid pixels does **not** guarantee that every contributing 128×128 model tile is valid. A point can receive a tile extending up to 127 pixels from it in either axis. A Euclidean distance threshold also fails to cover square tile corners. The most faithful stricter diagnostic is to enumerate the actual evaluated contributing tiles and retain a pixel only if **every contributing tile has complete declared geometry/source support**. Alternatively, a 127-pixel Chebyshev-neighborhood support test is conservative. Compute distances with an explicit false border/padding, so crop edges participate even when the whole local mask is true. Label a weaker 64-pixel mask as a boundary-margin heuristic, not full model-context protection.

The runner mask affects scheduling and final validity; it does not remove invalid pixels from raw-patch normalization. Thus partial-support tiles can change robust-MAD normalization and predictions at nominally valid pixels. The current [predeclared method](PREDECLARED_METHOD.md) retains the fixed 64-pixel primary interior and a secondary strict mask excluding the support of **any scored tile containing invalid geometry or a pixel with no observed intensity across the selected 17 depth layers**. This selected-window source-support rule must not silently become an all-31-layer criterion: the two are different diagnostics. All-zero intensity still does not establish archaeological absence or an ink-negative label. Apply the exact stated criterion consistently, use common masks for controls and report retained counts. Keep stored raw scores unchanged; do not blend a new map only from favorable tiles after seeing a signal.

## Frozen larger-crop inference and depth controls

Choose the crop from a finite, documented set of 128-aligned 1536 or 2048 windows using geometry coverage and fully valid model-tile support. Freeze size preference, ranking, tie-breaking, minimum coverage and crop source coordinates before native source fetching/inference. If all required criteria fail, report that outcome and record an amended design before opening predictions. Every candidate need not be physically independent; preserve the overlap description.

Retain all 31 genuine native layers and verify the chosen export's rendered offsets, normal flip, slice step and volume mapping. The small segment's 93 exact CT intensity matches only establish its tested neighborhood; they do not globally validate a different mesh/export. Where correspondence remains uncertain, a few predetermined, geometrically separated native CT profile checks across the new crop can test source/render consistency without selecting detector hotspots.

Use both pinned final-step seeds at **7:24 forward** as the larger-field primary run. Preserve the same 17×128×128 model input, upstream patchwise robust-MAD 1/99, CPU float32, stride 64 and Hann blending. Expanding the canvas must not expand the model patch or pad invented depth layers. The root's bounded control plan uses a **fixed central 1024×1024 context**, with local origin [256,256] for a 1536 crop or [512,512] for a 2048 crop. These origins preserve the 64-pixel tile phase. Run seed42 controls **5:22 forward**, **9:26 forward** and reversed **7:24** on that same predefined context. Compare against seed42 forward primary, verify the central-context forward map agrees with the larger-field map on their fully shared interior, and state that the controls cover only this subregion. They cannot validate leads outside it. Documented forward geometry remains the primary convention; never select whichever control resembles text most closely.

Compare depth controls on the intersection of valid/evaluated/source/boundary support across all compared runs; otherwise mean changes or correlations can arise from changed denominators. Report exact source indices, mean absolute score change and correlation where defined, together with image and geometry review. A depth-sensitive hotspot is a detector behavior observation, not ink confirmation.

The existing bounded runner's 1024-pixel dimension guard requires a separate reviewed adaptation before these larger inputs can run. Changing only the canvas bound and retaining identical inference functions is a reasonable source-preserving extension. Capture its startup code hash and use fresh output directories.

| Crop | Native chunk grid | Raw surface bytes | 128 tiles at stride 64, per seed |
| --- | ---: | ---: | ---: |
| 1536×1536 | 12×12 | 73,138,176 (69.75 MiB) | 529 |
| 2048×2048 | 16×16 | 130,023,424 (124 MiB) | 961 |

These bounds assume 128-aligned native crops inside the source canvas and retain all 31 layers. Two primary seeds schedule 1058 or 1922 tiles before geometry/empty-source skips. Use the already measured CPU throughput for a disclosed estimate and update from observed timings; do not claim these arrays require new hardware. Four effective CPU cores and 32 GiB RAM are adequate for bounded FP32 inference with sequential seeds. Dense maps, masks and scores fit comfortably, but source hash verification and coordinate-tree queries should remain bounded.

## Interpretation and preserved review evidence

Retain per-seed float32 maps, evaluated masks, the geometry/source/boundary masks, original intensity views, fixed score-scale panels and fixed-coordinate depth results. Treat strict raw-score >0.5 as the same descriptive midpoint diagnostic under 0.25/0.75 BCE targets. Report support counts and model-response fractions, not authenticated-ink fractions. Two seeds share a training corpus; larger pixels, adjacent depths, segment extensions and overlapping CT slabs do not create independent trials or support pixel-based confidence intervals.

The second experiment's PHerc1667 control established current annotation agreement in a region whose physical segment/representation overlaps training; exact historical label bytes remain unproved. That control cannot supply PHerc0800 false-positive rates or calibrate target scores. PHerc0800 still has no trusted labels/vetted reading for these regions, so precision/recall, cross-scroll accuracy, recovered letters and translation remain unavailable. Image-supported coherent strokes and specialist review are the later scientific gate.

Sources inspected include [live target inventory](../live-target-inventory.json), [old full geometry summary](../../data/experiments/pherc0800-surface-geometry/full-surface-geometry-summary.json), [native CT correspondence](../first-experiment/native-ct-consistency.md), the old geometry/selection helpers, and [comparison design](../second-experiment/comparison-design.md). New selected-geometry files and frozen crop coordinates should be linked into the third-experiment provenance when acquisition completes. Only this design-review report was added by this review.
