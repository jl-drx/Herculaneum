# PHerc0139 native physical-context separation

All265 mask-frozen contexts pass the geometry contract. A conservative physical-volume overlap test removes17 training contexts when w035 is held out and15 when w039 is held out. Holding out w044 removes none. Every original evaluation core is retained. These exclusions establish separated represented input footprints under the declared native render geometry; they do not establish independent specimens, annotation accuracy or teacher exposure freedom.

| Held-out winding | Evaluation cores retained | Original training contexts | Training contexts retained | Retained training supervised pixels |
|---|---:|---:|---:|---:|
|w035|109|156|139|1,578,067|
|w039|54|211|196|2,112,553|
|w044|102|163|163|1,638,213|

The source cohort came entirely from the previous mask-only selection. No ink labels or ink counts were read for context selection or exclusion. The frozen protocol contained a nominal per-winding count typo; `COHORT_COUNT_ERRATUM.md` preserves the correction to109/54/102 and the authoritative CSV hashes without changing any selected ID.

The public catalog directly links each native rendered array to its transformed native-coordinate mesh on volume20250728140407 (9.362µm). Three metadata objects and nine coordinate TIFFs were acquired after the protocol freeze and conditional length/ETag checks, totaling **4,378,451 response-body bytes**, below32MiB. No CT chunk or model was fetched/run. The exact native renderer commit is530584554c031c982ba3aa243e1ea39c4bef9818. The retained original source receipt proves the copied native QuadSurface and renderer identities; the distinct d9a472e file is transform-source context, not the native rendering version.

For each256×256 input context, the audit evaluated every raster pixel using the source's staged-FP32 native geometry and a separate full-precision interpolation alternative. All17,367,040 context-pixel occurrences have valid geometry. Repeated overlapping pixels count repeatedly; this is not a count of independent observations. Unit normals passed the1e−5 norm tolerance. Full28-plane offsets are−13.5 through+13.5.

The primary XYZ boxes use the union of both base-field extrema expanded by14.51 voxels on every axis:13.5 for the largest possible component of normal displacement plus1.01 for trilinear neighbor support and arithmetic tolerance. This conservative rule encloses all represented depth rays without assuming their direction, and is intentionally looser than exact ray bounds. Exact endpoint-ray boxes enlarged by1.01 are retained only as checks. The largest observed staged-versus-full-precision coordinate difference is0.001315 voxel; no arithmetic-based patch replacement occurred.

There are46 intersecting cross-winding box pairs, all between w035 and w039. Box intersection establishes only possible input overlap; it does not prove actual common source voxels or tissue. The three fold rules reject any training context whose conservative box intersects any original held-out context. The preserved per-fold files list every retained and rejected ID and reason. Training within one side can contain overlapping patches; the audit concerns cross-fold separation.

Six synthetic fixture groups passed before real boxes: flat/tilted planes, raster centers, flipped normals, all28 rays plus interpolation-neighbor enclosure, touching/disjoint boxes, and fold exclusions. Independent review is retained in `review/`; consult its final status before claiming independent completion. `RESULTS.json` contains all265 boxes,46 pairs and three folds; `COMPUTE_FREEZE.json` pins inputs and implementation. All previous source-alignment failures and label provenance qualifications remain unchanged.

Next admission questions remain actual CT availability/support, fixed normalization using admitted training inputs, valid training-only teacher transfer if distillation is tested, and a defined common annotation endpoint. This audit does not launch training, remove ancestry uncertainty or authorize GPU use.
