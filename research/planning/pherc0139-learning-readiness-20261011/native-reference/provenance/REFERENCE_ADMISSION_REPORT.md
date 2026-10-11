# Native PHerc0139 reference admission

11 October 2026. **Current public native labels can support a useful annotation-agreement development endpoint. They do not require the unresolved fine78-to-native registration.** A fresh native-input learner with whole-winding exclusions is a more direct next learnability test than training on the tiny w040 teacher-sensitivity packet. The released models have already trained on these windings, so their scores would be descriptive comparators, not untouched-test performance.

This audit read only retained catalog/source text and public metadata: 28 metadata GETs totaling53,763bytes, one teacher HEAD, and six label/supervision shard HEADs. No label/prediction pixels, new CT, inference, training or paid compute. Exact responses and public URL/ETag/length ledger are retained. No additional metadata acquisition is needed for the mask intake.

## Finite admitted candidates

| Winding | Exact public segment | Native CT ZYX / label YX | Catalog ink pixels | Catalog supervision pixels | Supervision/canvas |
|---|---|---|---:|---:|---:|
|w035|20260317000000-w035_2026031718|28×5820×5240 / 5820×5240|396,164|1,071,121|3.512%|
|w039|20260302000000-w039_2026030210|28×8560×7720 / 8560×7720|184,948|576,220|0.872%|
|w044|20260115000000-w044_2026011522|28×6040×8160 / 6040×8160|455,469|1,190,307|2.415%|

All three use volume20250728140407, native113keV/9.362µm. Current S3 label set20260918 is under each segment's `ink-labels/9.362um-volume-20250728140407/20260918/{inklabels,supervision}.zarr`. Live level0 label/supervision shapes equal native CT XY exactly. Their root metadata declares extraction of original HF plane14, unchanged pixel values/canvas/geometry, and9.362µm XY spacing. Plane14 is a storage/layout convention, not independent physical ink depth.

This is stronger than matching dimensions alone: declared source array, plane, target volume, render path and re-encoding operation agree. It is still a publisher coordinate contract, not an independently measured per-pixel alignment certificate. The intake should verify array values, masks and acquired-byte hashes; an eventual model study should verify actual selected CT support. Do not apply the w040 fine78 source-frame correction to these native-frame labels.

`CANDIDATE_MANIFEST.json` contains exact input/reference URLs, metadata pins, ancestry caveats, aliases and missing checks. Counts in this table come from metadata; this agent did not independently count pixels or establish the ink/supervision intersection.

## What the source history permits

The catalog calls these native annotations made on volume20250728140407, with no cross-volume transfer in this native republication. Pinned public training code uses the original five native sets as sparse manual supervision, calibrates the fine78 teacher against them, and preserves manual precedence. They are not the numeric fine78 prediction TIFF being proposed as our pseudo-target.

However, the current source HF arrays explicitly record `native-rerun`, private `inklabels-9.362um.report.json` paths, package`0e90a7cce (tifxyz-label-transfer-fixes)`, seam anchoring and cleanup. Their audits say strict=relaxed, zero gate rejections and zero blob removals, but11/3/7 ink-hole pixels were filled on w035/w039/w044. The resulting counts equal the catalog counts. We cannot reconstruct the private original transform/review history from this metadata. The catalog's phrase 'no transfer involved' should not be generalized to absence of all historical resampling/cleanup.

The September crop review explicitly covered the *aligned fine rasters*, not a separate review of these native arrays. Original annotator identities, whether annotators consulted predictions, and complete fine78-teacher training ancestry remain unverified. Therefore use **agreement with current released sparse native annotations**, not authenticated ink accuracy or fully independent human truth. Those limitations do not block a declared development comparison using a fixed reference for all arms.

All three windings appear in official native sparse training and Nieuwlaar's matched control/treatment configurations; all three also have aligned fine representations. w044 is the physical winding behind training alias`pherc0139-w028`; catalog publication notes two annotations`0139/w028_20260115221` and`0139/w044_2026011522` of one wrap. Group all such aliases together. The three manual sets also participated in fine78-threshold calibration, so do not call them fresh calibration-independent tests of that historical hard-target rule. A new scratch learner can exclude them by whole group, but this does not erase possible teacher ancestry in a later distillation experiment.

## Efficient bounded intake

`NATIVE_REFERENCE_INTAKE_PLAN.json` is ready for the parent intake implementation. Each level0 array is one Zarrv3 outer shard at`0/c/0/0`; its inner512×512 chunks are compressed with Blosc/zstd. There is no need for thousands of blind HF chunk requests.

- All three supervision objects: **29,017bytes** total.
- All three label objects: **36,631bytes** total.
- All six together: **65,648bytes**.
- Largest sequential decoded full mask:66,083,200uint8 bytes. Crop outside-array shard padding away.

Whole-object GET with retained`If-Match` and exact expected length is simpler than range access for these tiny objects. Verify the index's CRC32C, index lengths/offsets and decompressed inner chunk lengths; missing entries are fill zero. Preserve each source SHA256 and the decoded-array hash. Supervision zero means excluded from scoring/training, never a negative class.

Read supervision masks first. Freeze a deterministic complete grid of eligible cores and a bounded subset, if needed, based only on supervision extent/count and native geometric/context feasibility, before decoding labels. A central CT-only crop is a poor plan because supervision occupies under4% of each canvas; its hit rate is unknown, not equal to the overall fraction. Do not replace a selected core because its labels are inconvenient or blank. Report conditional class counts and any unscorable core after label access. The root's separate intake protocol owns the exact selection rule.

## What remains for the physical split

Different winding IDs are a useful nominal grouping, **not proof that their full CT input contexts are disjoint**. The28-plane native window spans about262µm of thickness, and slabs around nearby wraps may overlap. Global3D bounding boxes overlap widely and cannot resolve this.

Before calling the split physically independent, compare the selected native-frame mesh positions/normals and complete receptive-field plus depth supports across folds; group aliases and shared source footprints. No random pixel or overlapping-patch split. If exact context exclusion is unavailable initially, report a **whole-winding development split with unresolved input-context overlap**. That still supports a bounded learnability diagnostic, but it does not establish clean population generalization or authorize a frontier-performance claim. Do not discard the full depth window just to make a split appear independent.

## w024 fine59 remains a useful secondary candidate

Exact prediction asset:

`PHerc0139/segments/20260215000000-w024_2026021572/ink-detection/PHerc0139-20260215000000-1.129um-0.22m-59keV-volume-20260413113053-L1-20260709123958-mrg20736-1um-s1z2-tile256-stride128.tif`

Its public metadata establishes a different scan and model identifier from the fine78 teacher: volume20260413113053, model20260709123958 / `mrg20736_1um_s1z2`, declaredResNet3D-152, v43 workflow, source layers[27,89), force-reverse=true, tile256/stride128. No evidence here establishes independent model-training ancestry or manual truth.

A significant convention: the nominal CT volume is1.129µm but this asset renders source group1. Live render metadata says effective **2.258µm in all three axes**, shape[116,29350,27400]. The stored`canvas_size` order must not be substituted for Zarr YX shape. Native CT is[28,7100,6620]. Whole-raster resize is the published benchmark convention, not a demonstrated fine59/native physical mapping. The catalog's0.366539 bounding-box overlap proxy is not a valid-pixel mask. The fine59/native regridded mesh pair needs its own source-frame and local-support qualification; the w040 correction is not a universal shift.

Use this later as alternate-acquisition/model reference agreement after fixing its mapping and valid support. It is not a prerequisite for the **native annotation-agreement** baseline, and its incomplete geometry should not keep that simpler baseline waiting. Title/w024 are excluded by the pinned native recipe and author reports, but the benchmark was repeatedly used for model comparison and Hecate ancestry is unverified.

## Recommended finite next learning question

**Can a small learner trained on current native CT and current native annotations recover annotation-associated signal on another winding?** This bypasses teacher transfer entirely and establishes a native baseline before a soft-target intervention.

Use the frozen intake cohort; perform three leave-one-winding-out folds, keeping aliases together. Use a fresh modest full-depth convolutional learner and a fixed low-capacity native-input baseline. Model and optimization choice must be frozen before fold evaluation; do not initialize from the released models trained on these three windings. Retain all28 input planes and the original mask semantics. Keep training-region exposure and loss fixed across comparisons.

Measure training-only optimization curves to an explicit plateau criterion/cap, then evaluate one frozen endpoint on each excluded winding, reporting each region/fold. If more capacity is needed to fit training labels, diagnose it using training-only evidence once; do not let an intentionally underpowered null decide whether native CT has information. Average precision and AUC here mean annotation agreement. Report class support and prevalence; constant-prevalence and simple native-feature baselines prevent a trivially good-looking score. Do not score unannotated pixels as negatives or estimate confidence intervals by treating pixels as independent.

**Decision-changing result:** reproducible improvement over the simple native baseline on the excluded winding in each fold, with adequate training convergence and a completed context-overlap audit, supports preparing matched pseudo/soft supervision on additional training tissue. A positive result without the overlap audit remains development evidence. Failure to fit training labels means representation/optimization is unresolved; successful fit with failed cross-winding transfer means coverage/reference/generalization is the next issue. Neither null proves ink absence. Do not add a new depth/fusion adapter in response.

The exact model, plateau/cap, practical performance margin and seeds belong in the next frozen training protocol after the mask intake supplies actual class/support counts. Choosing numerical power claims before knowing those counts would invent precision. Current authorization permits preparing that concrete protocol; this metadata-only review did not run it.
