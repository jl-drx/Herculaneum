# PHerc0009B surface contract — 8 October 2026

**Partial resolution: the intended offset operation and renderer geometry are established, but the actual published transfer execution, annotation depth and mapping-validity release are unresolved.** This is a source-backed contract audit, not an acquisition or model run. No candidate arrays or predictions were inspected. Standing ordinary-research authorization persists; a scientifically frozen acquisition/inference contract has not been created.

The exact candidate is PHerc0009B fragment, segment `20250919125754-auto_grown_20250919055754487_inp_hr`, target volume `20250820154339`, published render `2.401um-0.35m-77keV-volume-20250820154339.zarr`. This identity remains outside our current adaptation, preservation teacher targets and calibration. Absence from the four originals' explicit 29 configured training roots does not establish absence from upstream teachers or pretraining.

## Offset operation: source correction, target output

The [official transfer library at commit 3ea17f54](https://github.com/ScrollPrize/villa/tree/3ea17f54a9b3d5fd1aaf73e1d2c8386dbaa9f30e/volume-cartographer/scripts/tifxyz_label_transfer) provides a concrete operation:

- `transfer.py:364` reads `--label-canvas-offset DY DX` in **full-resolution source-canvas pixels**, then scales it to the source label raster's dimensions.
- `core.py:1364` defines label pixel `(i,j)` as depicting source-canvas position `(i+dy,j+dx)`.
- `core.py:1759` gathers the source category at `floor(mapped_source_position × label/source_size − offset)`. It writes that category at the **target-canvas output index** (`core.py:1792`). The same geometry and offset apply to the additional supervision/validation masks.

Thus the intended offset is baked into the transferred result. A correctly transferred published output needs **zero additional canvas-offset shift**. Applying `[-38.351,-44.944]` again would repeat a source correction on the target output. The documented full-resolution units also rule out treating those numbers as level-2 pixels. The residual values and the review's `[-18.5,-25.0]` component are evidence about the offset construction, not additional corrections to add again.

**Actual execution is not fully verified.** All three current group attributes identify transferred work TIFFs and private scratch `.report.json` paths, record the same offset, and point to `provenance.json`. Their OME metadata contains scales only, with no translation or explicit “applied” flag. The actual report/provenance is absent from the complete release prefix. The declared package revision ends `-dirty`; its clean base `bfef6abe...` exists in Villa, but lacks this transfer-library directory. The officially documented `merge-ink-pipelines` branch contains the inspected library. Its current public `make_label_zarrs.py` produces **Zarr v2, 65 planes, label plane 32**; the actual release is **Zarr v3, 2D indexed shards**. That helper is not the historical publisher and its placeholder label plane does not prove the original annotation's CT depth.

This closes the intended sign/unit/application semantics, but not the historical caller's actual source/target mesh, affine, offset arguments and dirty modifications. `affine=null` in the catalog is not promoted to a proved identity transform. A future conditional public-reference comparison could explicitly accept the publisher's canonical-canvas declaration; it must retain this execution limitation and cannot claim independently measured alignment.

## Current masks and missing mapping validity

The current S3 prefix inventory contains **39 objects**, is explicitly **not truncated**, and contains exactly `inklabels.zarr`, `supervision.zarr` and `validation.zarr`. No mapping-validity array, public report or `provenance.json` occurs in that published prefix. No alternative public validity path is declared in the inspected sources. The retained catalog says `validity_published=true` and describes published mapping validity; that claim contradicts the observed release inventory.

| Verified metadata | Result |
| --- | --- |
| CT level-2 ZYX | 109 × 6,340 × 5,130, uint8, uncompressed 109 × 128 × 128 chunks |
| Ink/S/V level-2 YX | 6,340 × 5,130, uint8, Zarr v3 |
| Full label canvas metadata | [25,360,20,520] **YX** |
| Full CT canvas metadata | [20,520,25,360] **XY**; these agree after axis interpretation |
| CT level-2 spacing ZYX | [2.401,9.604,9.604] µm |
| Ink/S/V OME level-2 spacing YX | [9.604,9.604] µm; scales only |
| Single level-2 ink/S/V shard lengths | 17,784 / 7,830 / 6,538 bytes; total 32,152 bytes |

The three metadata descriptions have matching canvases, role-specific rasters and the same offset, supporting intended common registration. This is not an array-level or independent geometric verification. Actual binary label/participation encodings and participating classes remain unobserved. The catalog's validity values are declared as 0 unmapped, 128 seam-filled, 255 measured; those values were not read from an actual released validity array. **A measured-only mask cannot currently be constructed from this release.** S/V participation must not be relabeled geometry validity. Any alternative participation policy belongs in the next frozen scientific design.

## Render depth and normal: resolved geometry, unresolved annotation

The archived [renderer at commit 53058455](https://github.com/ScrollPrize/villa/blob/530584554c031c982ba3aa243e1ea39c4bef9818/volume-cartographer/apps/src/vc_render_tifxyz.cpp) matches the retained export record. It records group 0, scale 1, 109 slices, `flip-normals=true`, XY `flip=-1`, rotate 0 and composite false. The published source attributes explicitly state `slice_step=1`.

`vc_render_tifxyz.cpp:340` defines `offset(k) = (k − (N−1)/2) × slice_step`. For this render the nominal zero-offset plane is therefore **zero-based source index 54**, established by the algorithm and settings. The sampler uses:

`P(k) = B + (k−54) D`, where `D` is the normalized negative geometric surface normal.

Its signed distance is `(k−54) × 2.401 µm` along the local flipped normal. This is not a fixed native-CT Z displacement. The position uses float32 bilinear surface coordinates; the normal uses the nearest cached mesh-node normal, followed by the recorded flip and normalization. CT sampling uses trilinear interpolation, clipping and uint8 truncation. `accum-type=max` alone is not a maximum projection: the archived default accumulation step is zero. The export record omits an explicit accumulation-step argument, so no hidden override/full-argv recovery is claimed.

**These facts do not locate the ink annotation at source plane 54.** The reference is 2D. Its source overview filename ends `_2um_max_22_42.tif`, but no retrieved original render manifest ties those numbers to this 109-plane export, a normal direction, a projection depth interval or an annotation-center plane. File naming is not a depth contract. Annotation plane, projection range, pooling window and prepared label plane remain unset.

Four-slice pooling also has a real phase choice. A block starting at source index `a+4j` has center `a+4j+1.5`; no mean of four consecutive integer indices can center exactly on integer index 54. Borrowing our previous `[13,97)` / label-plane-10 recipe would put its center at **54.5**, +1.2005 µm along D. That arithmetic illustrates the unresolved phase; it does not select or authorize that recipe for this fragment.

## XY mean versus categorical-decimation phase

Labels declare repeated `[::2,::2]` decimation. Exact archived `Zarr.cpp:212` instead builds CT XY pyramids with repeated 2 × 2 means, preserving Z, and rounds each level as `(sum + count/2) / count` using integer division. The level-2 CT should not be reconstructed as one unrounded 4 × 4 mean.

In the interior, label level-2 index `(i,j)` takes full-resolution category `(4i,4j)`. CT level-2 index `(i,j)` averages support `[4i,4i+4) × [4j,4j+4)`, whose kernel centroid is `(4i+1.5,4j+1.5)`. The centroid difference is **1.5 original pixels per axis = 3.6015 µm in nominal canvas units = 0.375 level-2 pixels**. Its native-3D displacement depends on the mesh; no mesh-array audit was performed. This is a representation-kernel phase, not a measured registration residual. Label OME omits a translation and CT OME declares zero translation; neither records this kernel-centroid phase. No automatic shift, resampling or fitted transform was chosen. A subsequent protocol must explicitly preserve or resolve this public representation convention and qualify physical registration accordingly.

## Remaining concrete gates

The minimal missing evidence is the actual transfer report/provenance with source and target mesh identities, affine and offset execution; the canonical mapping-validity array or a revised participation claim; and the original annotation/render depth provenance sufficient to choose a model window and pooling phase. A later frozen label-only acquisition can verify binary encodings and class support, but cannot recover these missing historical facts by itself. The catalog's “annotated_on=unknown”, model-prediction-based review and dirty publisher remain annotation/independence limits.

The audit used **20 HTTPS attempts / 318,867 response-body bytes**: 19 HTTP 200 and one retained HTTP 404 for the absent clean-base library path. Verified TLS was used; no redirects or retries were followed. Only official metadata and source text/code were read. No scans, images, label-array payloads, predictions, tensors, models, accounts or paid resources were used. Public CT is CC-BY-NC 4.0; modern annotation terms remain unresolved before redistribution. Prior files remain unchanged.

[Exact facts and unresolved fields](surface-contract.json), [source hashes and pinned blob checks](SOURCE_RECEIPTS.json), and [all request receipts](request-ledger.json) make the result reviewable. The independent renderer read-only review confirmed its formula and limits; source files were checked unchanged during report construction.
