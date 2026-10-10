# Regional geometry and native-render correspondence

**All 284 fixed native-CT predictions exactly reproduce the historical published rendering. No shared optical/CT non-ink landmark was accepted by either the requested agent or the root reviewer.** These results strengthen confidence in the CT input mechanics while leaving optical registration and anatomical ink depth unresolved.

The panel contains the same four previously model-scored Frag6/PHerc51 development contexts. It is not a fresh ink benchmark. Target rendering values were already cached; new native source bytes provide a regional replication of the historical renderer test. Geometry sites and source targets were fixed before their acquisition/comparison.

## What was tested

- Cached display packet: all 63 layers for each crop, exact optical crops, and fixed raw row/column 193 cross-sections. CT displays use 3×3 spatial means at fixed 0–65535 scale, optical retains 0–255 and its original alpha. No labels or model predictions enter this packet.
- New geometry: a fixed 3×3 lattice per crop, with a 3×3 PPM neighborhood at each site: 36 sites, 324 records, 108 byte ranges. All records are mapped with unit normals; all 63 declared sampling depths at every centre are in source bounds. Local tangent scale and shear vary, so a single physical pixel scale is an approximation. Normal/raster cross-product sign is not an anatomical side determination.
- New native source: all 63 depths at each crop centre plus layer 32 at each of the other 32 sites: 284 comparisons across 40 new native chunks. Primary criterion was exact reproduction using the previously identified v2.26.0 historical interpolation corner defect and nearest-even uint16 rounding. Correct trilinear interpolation was retained only as a paired reference.

| Context | Historical exact matches | Historical max error | Correct-trilinear exact matches | Correct-trilinear MAE |
|---|---:|---:|---:|---:|
| r544-c4128 | 71/71 | 0 | 4/71 | 508.099 |
| r2080-c3616 | 71/71 | 0 | 3/71 | 370.521 |
| r4640-c1056 | 71/71 | 0 | 2/71 | 289.408 |
| r6688-c1568 | 71/71 | 0 | 2/71 | 347.563 |

Errors are uint16 digital units. Correct interpolation differs because it does not reproduce the historical defect; it is not a worse estimate of native CT. This test does not justify silently changing the inputs expected by the existing checkpoint.

## Verification and visual review

Thirteen meaningful fixtures passed (five strict transport, four geometry, four interpolation/rounding); zero failures, errors or skips. Separate arithmetic reconstructed 4,128,768 displayed CT means, 2,359,296 optical channel values, 193,536 cross-section values, all 324 PPM records and 396 geometry scalars. Maximum geometry difference was 4.44e-16. There was no overlap with old cached PPM records.

A second native decode pass used scalar PPM reconstruction, direct byte-addressed voxel extraction and nested interpolation, separate from the worker's weighted corner formula. It checked all 2,272 corners and 284 targets; all rounded historical values matched exactly. Maximum unrounded difference was 1.46e-11; the closest worker value was 0.001232 digital units from a half-integer. Numerical verification shares the same codec and source bytes; it is not an independent physical measurement.

At the user's request, an agent inspected every displayed CT layer and the optical crops without being supplied root's interpretation or model scores. It found zero accepted or tentative landmark pairs. Root then inspected all four enlarged panels and agreed. Optical crops remain diffuse in lossless exports; a shared unique tear, edge, fold or intersection cannot be localized. This is a separate visual assessment, not a blinded accuracy experiment. See `ROOT_VISUAL_REVIEW.md` and `visual-review-v1/REVIEW.md`. An absence of identifiable landmarks does not prove misregistration.

## Resources and limits

44 source requests completed without redirects or retries, with TLS verification: one PPM request and 43 native/metadata requests. Returned bodies total 160,397,866 bytes; reservations total 168,068,972 bytes. The native worker took 12.634 seconds with 47,226,880 bytes peak RSS. Independent native verification took 0.403 seconds with 38,248,448 bytes peak RSS. Each of 40 native chunks was decoded once by the worker and once by verification (320 MiB combined decoded work). No full scan was downloaded.

No model forwards, optimizer steps, GPU jobs, R2 uploads, publication or submission occurred. GPU spend remains £0; the first-pilot total cap remains £10. The repository remains at 49fabf3e890d5b880f391b27d581f1e7a1ebdbcc with only the preserved untracked `.codex/` status. Other reserves remain untouched. Maintainer request remains last-confirmed not sent, with no supplied reply.

## Research decision

Retain the baseline checkpoint and the phase32 negative averaging result. Do not fit an optical warp or pick a depth from label agreement. Regional CT source/render fidelity is now well supported at the tested sites; this does not establish independent ink accuracy, optical truth, or a physical ink surface.

Proceed to a small CT-only structural depth-reference prototype, first on synthetic known transitions and then descriptively on these cached stacks. It will detect a change in spatial texture energy, with abstention for flat or competing transitions. Its output is a texture boundary proxy, not an ink depth. Freeze its rule before real-stack evaluation, retain every outcome, and do not tune it against label/model performance. This advances a possible depth-conditioned Plinius input while keeping the scientific claim testable and limited.
