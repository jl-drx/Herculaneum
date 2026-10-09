# PHerc0800: measured mesh-to-M7 disagreement

A fixed 9×9 node lattice on each of the three previously ink-tested meshes selected 243 locations before M7 array values were read. 104 had finite nondegenerate mesh normals; invalid nodes were retained in the denominator, not replaced. Sampling used the pinned author's normal calculation, level0 coordinates and nearest-neighbour M7 values at 31 integer normal offsets −15… +15. 71 observed public chunks supplied 30,129,968 compressed bytes. There were no missing chunk objects and no fabricated background values.

The diagnostic measures the centre of the nearest positive contiguous run in the published binary M7 prediction. It does **not** apply snapping, neighbourhood interpolation or smoothing. Equal nearest runs remain ambiguous; a run touching a search endpoint would be censored. This preserves the measurement rather than filling it from neighbouring sites.

| Segment | Selected / usable normals | Measurable / tied | Median absolute offset | 90th percentile | Measured sites >2voxels |
|---|---:|---:|---:|---:|---:|
| 20251029010146 |81 / 41|41/0|0.5|2.5|6/41|
| 20251028222030 |81 / 32|30/2|1.0|4.1|7/30|
| 20251028220955 |81 / 31|31/0|1.0|3.5|7/31|

None of the three conditional medians exceeds two native voxels.20 / 102 measurable sites do;57 / 104 usable normal profiles contain multiple predicted runs. The latter count does not prove 57 wrong-sheet assignments: it shows that neighbouring predicted surfaces are often present in the search slab. The two tied cases are explicitly unresolved. No nearest run was endpoint-censored in this sample.

This sparse lattice covers the source meshes, not just the old detector hotspots. It is not weighted by tissue area, a dense whole-surface assessment or a random sample of all PHerc0800. Normals and nearest-voxel sampling introduce discretization; smooth faces, sheet centres and ink-bearing surfaces are different objects. The binary M7 threshold is part of the released artifact. M7 is a model prediction and may share ancestry with the geometry. Agreement cannot authenticate the true writing face.

Decision: do **not** apply a blanket recentering or commission a new surface model because of the historical two-voxel rule. Preserve a local-outlier/ambiguous subset for CT cross-sections and compare original versus carefully snapped geometry where warranted. A 66-layer render and±15model-centre sweep remain useful because the original target controls barely moved the window and M7 agreement does not locate ink within the sheet. The original engineering trigger required CT-annotated placement error; this test has measured disagreement with M7 instead, so it does not settle that independent-accuracy gate.

`geometry/RESULTS.json` retains all 243 locations, normals, raw run counts and statuses. `geometry/VERIFICATION.json` independently reconstructs all 104 profiles from retained compressed chunks and reclassifies runs without using the measurement helper.
