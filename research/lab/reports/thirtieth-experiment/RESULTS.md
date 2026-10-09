# Phase30: a renderer indexing defect explains the CT mismatch

We reproduced the published Frag6 surface values by accounting for a specific interpolation defect in the version named by its TIFF metadata. The source-derived calculation matched all **189 already inspected values exactly**, then matched **three predictions frozen before reading a new layer**. This is concrete progress on the source-to-render contract. It establishes neither ink accuracy nor a complete optical reference.

## The experiment and its preserved negative result

We used the nine potential comparison sites selected in phase21. Two were already unmapped and remained excluded, without replacement. The seven complete sites supplied 3×3 neighborhoods on published layers 0, 32, 64:81 calibration pairs at the first three sites and108 check pairs at the remaining four. Selection depended on the retained geometry, before CT values were opened.

The initial frozen comparison tested both normal signs with trilinear, floor and nearest-integer sampling from the named53 keV Zarr. Positive-normal trilinear was selected on calibration data and strongly outperformed the alternatives, but **failed** the strict raw-value reproduction limits. Its check MAE was 215.732 counts, maximum 2343.237; the same recipe at deliberately shifted coordinates had MAE 7876.225. All check sites had informative intensity ranges. Five decision tests passed, including guards against selecting a recipe on check outcomes.

The unchanged one-pass sampler read 14 native chunks once. A separate root-owned scalar implementation rebuilt every point/corner and TIFF byte address from the retained bytes; all 1,134 sample values matched exactly, maximum difference 0. This ruled out disagreement between those numerical implementations. The original `MAPPING_NOT_QUALIFIED` result remains sealed and was not changed to a pass.

## The source-informed explanation

Each original TIFF has an uninspected 27-byte Software-tag payload. Reading those exact pinned ranges identified `volume-cartographer 2.26.0` in all three layers. The matching public tag resolves to commit[`7640ba6c8e41f39a7aef63e9780cd13c296b2d15`](https://github.com/educelab/volume-cartographer/tree/7640ba6c8e41f39a7aef63e9780cd13c296b2d15), tree `1d8cc0c9959ae3ca4cb42a78bd940fe1ef6b7fc6`.

In that version's [`Volume::interpolateAt`](https://github.com/educelab/volume-cartographer/blob/7640ba6c8e41f39a7aef63e9780cd13c296b2d15/core/src/Volume.cpp#L174), the `c10` term uses the `(x1,y0,z0)` corner where standard trilinear interpolation requires `(x1,y1,z0)`. It reuses one corner and omits another. Relative to correct trilinear interpolation, the unrounded difference is:

    dx × dy × (1 − dz) × [I(x1,y0,z0) − I(x1,y1,z0)]

The inspected line-generator implementation calls that interpolator at `position + normal × offset`. The layers application exposes radius, interval, direction, source-volume and optional transform settings; the original invocation is still unknown. The code and GPLv3 license are retained for provenance. No downloaded C++ was compiled or executed, and no application/baseline source was modified. Our local algebraic replay is confined to historical-artifact compatibility research.

With the original positive-normal recipe `PPM_XYZ + (layer−32) × normal`, applying this source-derived correction and nearest-even rounding matched **189/189 uint16 values exactly**. The replay's saved array has the same SHA256 as the published189-value array. This was explicitly a **post hoc explanation**, with no fitted intensities, coordinates, offsets or weights. It is not fresh confirmation. The nearest half-integer was 0.002108 counts away, so the observed matches were not ambiguous half-rounding ties.

## Fresh-layer check

Before requesting any bytes of layer 16, we froze predictions for the center point at the first three original check sites, using cached native data at `PPM_XYZ −16×normal`. The prediction manifest SHA256 was`e42095d722466e5f6adfcde66c8b412d0aacd0ccbcfd3a025fb154e672acdad3`. Seven bounded requests then checked that TIFF's header/IFD, acquired its three target values and confirmed the same Software tag.

| Raster(y,x) | Frozen historical prediction | Newly read value | Correct trilinear, rounded |
|---|---:|---:|---:|
|4426,4963|31274|31274|31556|
|7081,1240|13069|13069|13189|
|7081,3102|44471|44471|44466|

All three matched exactly; all three distinguished the historical behavior from correct trilinear by at least 2 counts. This is fresh target-layer confirmation of a specific rendering behavior on three points. Native chunks had already been acquired, and these are not independent physical parents. No ink labels, optical images or model outputs entered either the explanation or confirmation.

## What this permits—and what remains

The previously opaque mismatch now has a tested explanation. We have narrow empirical support for the named native store, PPM coordinates, positive stored-normal ordering, layer 32 origin and unit-voxel layer offsets **at the sampled points/layers**, when reproducing the historical interpolation behavior. Correct interpolation should remain correct in our sampler; reproducing a historical artifact is a separate explicit compatibility mode. Never silently switch kernels between arms of the depth-window experiment.

This does not authenticate the producer's exact command, compiler/binary, every intermediate layer or every raster location. The version tag's commit date is 2024-08-09 while the server reports 2023 Last-Modified dates for the TIFFs; those metadata dates cannot be treated as an authenticated production chronology. The behavioral reproduction is stronger evidence than an inferred date narrative. No claim is made that this defect is newly discovered, persists in the current upstream version, or explains model accuracy.

The native volume descriptor's 3.24 µm pitch and Zarr's compatible unitless layout remain separately recorded. Anatomical inward/outward direction and ink placement are not established by matching CT intensities. Full panel coverage, supplied-label zero/unknown conventions and optical registration still need review. A supplied-label comparison can proceed under its stated convention once those panel inputs are bound; it must not be relabeled independent optical truth.

**Next:** construct the actual bounded reference panel using explicit rendering conventions, inspect supplied mask/label support prospectively, and qualify pixel correspondence before the fixed two-arm model comparison. The source ambiguity has narrowed substantially; another generic sampler test or blind interpolation search is unnecessary. See `NEXT_STUDY_DRAFT.md` and `REFERENCE_STATUS.json`.

## Resource and exposure record

| Part | Requests | Response-body bytes | Reserved bytes |
|---|---:|---:|---:|
|Primary CT comparison|80|56,276,149|58,917,546|
|Three Software tags|3|81|84|
|Version-specific public source|10|309,614|10,485,770|
|Fresh layer 16 check|7|193|200|
|Total|100|56,586,037|69,403,600|

No redirect or retry occurred. This is body accounting, not total TLS/header wire traffic. Fourteen source chunks contain 29,360,128 decoded native voxels; 56,272,328 compressed source bytes were retained. The primary acquisition/comparison took 43.859 seconds, peak process RSS 50,708,480 bytes. Source decodes were 14 primary + 14 scalar verification + 11 historical replay + 3 fresh prediction preparation = 42 decodes / 168 MiB decoded work. Only 14 unique chunks were downloaded. The source-informed offline extension was frozen before those additional decodes; it did not alter the failed comparison's acceptance rules.

Frag6 now has CT exposure for source/geometry validation: 192 published CT values inspected, plus the retained native chunks. Do not repeat the older “no Frag6 CT pixels” claim. Ink labels, optical pixels and ink-model scores remain untouched. Other reserved parents remain unchanged. There was no model forward, training, transcription, translation, R2 upload, paid resource, publication, submission or outreach. Maintainer request remains user-confirmed NOT SENT. GPU spend £0 against the £10 total first-pilot cap.

All work is outside `/workspace/Herculaneum`; repository HEAD49fabf3e890d5b880f391b27d581f1e7a1ebdbcc and pre-existing untracked `.codex/` are preserved. Root owns source, numerical and report review; no independent-agent validation is claimed.
