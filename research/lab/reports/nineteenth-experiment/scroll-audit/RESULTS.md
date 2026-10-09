# PHerc0009B and PHerc0500P2: phase 19 provenance and geometry audit

Neither fragment is ready to serve as an independently authenticated accuracy holdout. This audit closes several operational questions, but also finds concrete publication gaps: the public label packages omit the advertised mapping-validity arrays and linked provenance files. Public historical training configuration includes both parent aliases. They can remain reserved from our future updates, but that reservation cannot establish upstream independence.

The work used **24 public text/metadata requests, 1,748,669 response-body bytes**, with 20 HTTP 200 responses and four retained HTTP 404 responses. No redirects or automatic retries occurred. No scientific image, label, mesh, CT array or model payload was read; no inference, fitting, upload, credential action or paid resource occurred. Prior artifacts and the tracked repository were not modified. [Request receipts](request-ledger.json) retain each exact URL, time, status, response body and SHA-256. [Source pins](SOURCE_PINS.json) bind reused records; [the machine-readable matrix](eligibility-matrix.json) contains all conclusions and parent exclusions.

## What changed relative to the tenth-stage audit

| Question | PHerc0009B | PHerc0500P2 | Consequence |
|---|---|---|---|
| Actual public label roles | Ink, supervision and validation confirmed, all 2D uint8 | Same | The S/V metadata gap is closed; binary value semantics are still unverified without a source contract or data read. |
| Mapping validity | Complete public inventory has no mapping-validity asset | Same | The catalog's `validity_published=true` claim does not describe these public prefixes. A measured-only scoring mask cannot presently be constructed from them. |
| Linked `provenance.json` | HTTP 404; absent from complete inventory | Same | Exact transfer/publication command and dirty changes remain unavailable here. |
| Mesh-relative depth center | Index 54 of 109 planes | Index 58.5 of 118 planes | The second render has no sampled zero-offset plane. Neither fact authenticates ink depth. |
| Recorded acquisition energy | 77 keV | 111 keV; raw acquisition record also says 111 | The 110-keV filename discrepancy is identified as filename versus recorded acquisition metadata, with the same scan/volume lineage. |
| Historical paper-config membership | `9b`, including the exact candidate segment, in both train and validation lists | `500p2a` and `-1` in both lists | Absence from the separate 29-root 9-µm configs is insufficient for a clean upstream holdout claim. |

## Public package completeness and actual storage

Each exact `.../ink-labels/<resolution>-volume-<id>/20260918/` prefix contains **39 objects**, with `IsTruncated=false` in a single bounded inventory. These are the three published label roles, each with six level metadata files, six data shards and one root metadata file. Neither inventory contains `mapping_validity` or `provenance.json`. Direct requests for `mapping_validity.zarr/zarr.json` and the root-linked `provenance.json` returned 404 for both fragments. This establishes absence at the inspected prefixes at the recorded time, not absence from every private or alternative location.

The six root metadata records and four additional level-2 S/V records were fetched. Ink level-2 metadata was reused from the pinned tenth stage. All roles share the appropriate YX shape, uint8 dtype, zero fill, and 512×512 inner chunks within a sharding-indexed codec. Level 2 is 6,340×5,130 at 9.604 µm for PHerc0009B and 6,190×3,470 at 8.86 µm for PHerc0500P2. Their OME label transforms specify scale and **no translation**. Actual value sets, supervision-negative semantics and validation overlap have not been read or inferred from uint8 storage. A zero fill is not proof of a negative ink label.

The two complete inventories are retained as [PHerc0009B](sources/PHerc0009B-package-inventory.xml) and [PHerc0500P2](sources/PHerc0500P2-package-inventory.xml). The advertised measured/seam-filled/unmapped encoding remains 255/128/0 in the catalog and public transfer documentation, but the required asset is missing from these packages. In particular, the historical claim that all seam-filled positive labels are rare or absent does not justify treating every supervised negative as measured.

## Canvas offsets: implementation meaning established, exact application still unproven

At the named base commit `bfef6abe073510ee8b595da390ed1d639316abc8`, `transfer.py` lines 385–402 treats `--label-canvas-offset DY DX` as a correction to the **source** annotation canvas. It scales the full-resolution offset into source-label pixels. `core.py` lines 1435–1437 describes label pixel `(i,j)` as source-canvas position `(i+dy,j+dx)`; sampling subtracts that offset before reading the source category. The transfer writes labels on the **target** surface canvas. This makes the catalog's offsets consistent with inputs to transfer, rather than a second shift for a consumer to apply to the target labels.

The declared offsets are `(-38.351,-44.944)` for PHerc0009B and `(+32.4,+34.0)` for PHerc0500P2, in full-resolution source-canvas pixels. Their retained quality notes include additional prediction-alignment history. **Do not apply either offset again to the published arrays, or fit a new offset to improve detector scores.** The exact publication command, transfer report and any post-transfer realignment must establish their final meaning before a scientific acquisition contract is frozen. Matching target shape and absence of an OME translation do not independently prove registration.

There is a material code/version limitation: the reported publisher commit ends in `-dirty`. Its public base commit's [converter](sources/transfer-make_label_zarrs.py), lines 44–48 and 57–79, writes a **65-deep ZYX volume at `LABEL_SLICE=32`**. [The constants](sources/label_zarr.py), lines 19–21, confirm that convention. The actually published packages are **2D YX**. The unpublished wrapper or changes therefore matter. The base converter cannot authenticate how these artifacts were built, and its plane-32 convention cannot be assigned to their 109-/118-plane CT renders. Public tree and code availability do not repair missing dirty publication provenance.

## Depth geometry and source naming

The renderer is already retained at the exact recorded commit `530584554c031c982ba3aa243e1ea39c4bef9818`; its SHA-256 was recomputed against the fifth-stage pin. `buildOffsetList`, lines 337–353, defines offset `(z - (N-1)/2) × slice_step`. Lines 273–287 negate the unit normal when `flip-normals=true`. Both candidate render receipts specify that flag, source group 0, slice step 1 and their native pitch. Thus the signed mesh-relative coordinates are:

- PHerc0009B: `(z - 54) × 2.401 µm`, along the flipped normal. Plane 54 lies on the reconstructed surface.
- PHerc0500P2: `(z - 58.5) × 2.215 µm`, along the flipped normal. Planes 58 and 59 are at −1.1075 and +1.1075 µm; the surface lies between them.

These are renderer coordinates, not authenticated inward/outward anatomy or measured ink location. The annotation source overview names end in `max_22_42.tif`; a filename describing a projection does not authenticate a single annotation plane or its mapping to the new render. No depth crop, reversal, pooling rule or label center is assigned. Borrowing the old 109-plane `[13,97)` pooling rule would still be an unsupported new assumption.

For PHerc0500P2, retained scan `20250508033249` records 111.0 keV in the top-level scan metadata **and** `creation.metadata.metadata.tomo.acquisition.energy`. Its scan radix, processing prefix and renderer input path contain `110keV`. The target volume's derivation points to that same scan. Use **recorded acquisition energy 111 keV**, while preserving literal 110-keV names for source identity. This is not an independent beam-energy calibration. PHerc0009B similarly has rounded `0.3m` catalog names while the acquisition record and source paths specify a 350-mm detector distance; the metadata identity is consistent.

## Parent aliases and label ancestry

The [retained full records](PHerc0009B-full-retained-catalog-record.json) include all three PHerc0009B volume IDs and 19 catalog segments; the [PHerc0500P2 record](PHerc0500P2-full-retained-catalog-record.json) includes four volumes and 46 segments. `eligibility-matrix.json` enumerates every catalog scan/volume/segment ID and the known textual aliases. Reservation must cover the whole parent and any later-discovered aliases, including `9b`, `0009b`, `new_traces9b/david_9b`, `500p2a`, `train_scrolls/500p2a`, `sean_hiddenlayers/-1` and `0500p2/-1`. This catalog inventory is not claimed to exhaust all external historical names.

The independently re-read paper configuration pinned at Villa `e583fb67468f483fadd73d52e06f0ab0fe5ba813` includes the exact PHerc0009B candidate segment and the other 9b segment `20250910185200`, as well as `500p2a` and `-1`, in both `training.train_segments` and `training.val_segments`. Their per-segment records specify the base aliases and original paths. This proves configured representation membership in that pipeline. It does not establish a particular checkpoint's execution history, exact loss masks, or validation overlap. Root's separate upstream exposure audit handles the checkpoint/teacher chain.

The annotation package source rasters originate under `s3://philodemos/datasets/ink/ink_030927/0009b/...` and `.../0500p2/-1/`. Its `annotated_on` remains `unknown`, with transferred registration and review against an unnamed “independent ink prediction.” That wording does not establish model-independent human or optical truth. No author-level or pixel-level original annotation record is present in the fetched package. The recorded CT license is CC BY-NC 4.0; no authenticated grant covering redistribution of these modern annotation packages was found. This audit performed private source analysis and no redistribution or terms acceptance.

## Exact remaining contract before any candidate data read

The candidate reservation remains in place. No crop was selected. The unexecuted contract must resolve these fields before CT/label acquisition:

1. A source-to-target transfer receipt including source/target surface hashes, the dirty publisher code or exact commands, offset application, and any final alignment. No model-based offset search is permitted on a reserve.
2. The actual mapping-validity asset, or an explicitly justified alternative participation definition. Do not silently substitute `S>0` for `S>0 and mapping_validity==255`.
3. Documented categorical ink/S/V values and the meaning of unannotated versus supervised negative pixels. Any later label-only verification must reject unexpected values rather than reinterpret them after seeing model results.
4. An annotation-to-surface depth relation and fixed normalization, signed depth coordinates, layer selection and resampling. Native XY pitches differ; a single borrowed center/window is insufficient.
5. Whole-parent/alias exclusions and the permissible evidence claim, informed by the upstream lineage result. Both could support a deliberately labelled supplied-reference agreement study after geometry closure; neither is presently an authenticated independent accuracy reserve.
6. A bounded, prediction-blind sampling plan fixed before reading labels or CT: exact number and size of regions, eligible participation criteria, deterministic order/seed, nonoverlap including context halos, no performance-driven replacements, and all selected contexts reported. Only after the preceding fields are resolved can its coordinates, byte budget and acquisition requests be frozen.

If an independently annotated holdout is the objective, the concrete public-package defects and upstream config membership make it more efficient to pursue a candidate with documented optical annotation lineage. These fragments remain useful candidates for future controlled transfer or reference-agreement studies; this audit does not discard their scientific value.
