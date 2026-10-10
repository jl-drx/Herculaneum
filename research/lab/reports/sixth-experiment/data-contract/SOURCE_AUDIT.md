# PHerc1667/w029 data contract — 8 October 2026

**The exact public surface and corrected aligned-mask contract are verified.** The selected 256 × 256 crop supports a small developmental comparison against public annotations. It does not establish independent authentic ink accuracy, teacher independence, correct papyrus-face placement or a manuscript reading. Selection used metadata and annotations before source intensity or predictions. This study prioritizes scientific usefulness; its source permissions still apply.

## Actual source identity and pooling

Canonical scan-derived source:

`https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1667/segments/20251212185248-w029_20251212185248662_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr`

| Field | Verified contract |
| --- | --- |
| Annotation alias | **pherc1667-w029**, not pherc0139-w029 |
| Public segment | `20251212185248-w029_20251212185248662_flatboi` |
| Volume / scan | `20251217075048` / `20251204164204` |
| Acquisition | 2.399 µm, 78 keV, detector distance 220 mm; masked uint8 export |
| Public native render | ZYX **109 × 38,000 × 31,320**, uint8; 109 × 128 × 128 chunks, uncompressed, `/` separator |
| Actual acquired level | **2**: ZYX **109 × 9,500 × 7,830**; spacing **[2.399, 9.596, 9.596] µm**; same chunks, dtype and compressor |
| Curated aligned arrays | Ink labels, supervision and validation each **21 × 9,500 × 7,830**, uint8, Blosc/Zstd bitshuffle, chunks **21 × 128 × 128**, zero fill, annotation only at **Z = 10** |
| Exact preparation | Source depth **[13,97)**; float32 means of four consecutive planes, `np.rint` to nearest even, cast to uint8. Read the published level-2 XY pixels directly; no additional XY pooling |
| Resulting input | **21 × 256 × 256**, nominal XYZ spacing **9.596 µm** |
| Model input | Pooled **[2,19)**, 17 planes; corresponds to source **[21,89)**. Pooled label plane 10 averages source **[53,57)**, mean source index **54.5** |

The pool operator is retained from [official Villa e0bbb8b4 preprocessing](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/preprocessing/prepare_9um_isotropic_input.py). Its centered start is `ceil((109-84)/2) = 13`. The previous w013 acquisition was read first, but its training-control assertions are not copied indiscriminately: w029 has a separate validation mask, and legitimate validation labels can occur outside public training supervision.

Current source `.zattrs` declares 109 planes, a one-native-voxel slice step, ZYX axes and mean XY downsampling. Curated label `.zattrs` independently names the same public surface, level 2, source shape, depth slice, output spacing and central annotation channel. All three curated attribute bodies have SHA-256 **`1a749c0c9d290f24424d9018c0a98ed80a3abe4f99424a0f773d66c7c8774cd6`**; their `.zarray` bodies have SHA **`52e50d0371f434ea084ad8f1fb02f31a0e1c4cc11b6e2f250faf63febb8b024a`**. Source level-2 `.zarray` SHA is **`6e33b6b5d93db9cc803d66590a5a41efd960abeb46281f5595b6b7f1dc5a8b9a`**.

Archived official render provenance identifies renderer commit **`650bd56663efb7225fdba73fe267ba86b8e29368`**, 109 slices, scale 1, rotate 0, `flip=-1`, and `flip-normals=true` for this exact surface and volume. [The extracted record](archived-w029-render-evidence.json), SHA **`6542a825f98ad9881042174c9fec89064bb2806c3ee2a94136196c682b0a61b1`**, links the original retained gzip metadata SHA **`7ec8dcaa3000b9716724c26a1991ee416491d8a6b3615238ca90ae11540abdd3`**. This records export provenance, not an independent native-CT rendering verification. Increasing published source Z is retained without a fitted sign, offset or reversal.

The earlier paper aliases `l_2/l_5` are still not mapped to these modern public segment versions. This experiment explicitly uses the resolved public w029 identity. It does not inherit a paper holdout claim through an unresolved alias.

## Corrected annotation lineage and mask semantics

Canonical curated root:

`https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w029`

The actual attributes record `annotation_source_results=results-fixed`, a rebuild from `aligned-public-21slice-v2`, and a private scratch-path `open-data-training-manifest-v1.json`. Mapping validity is explicitly relaxed at ≥128, restricted to additions within 128 pooled pixels of strict supervision, then cleaned using stated blob/hole/island rules. The full historical alignment manifest and per-stroke human-versus-iterative/model provenance are not supplied by these attributes. Official curated documentation says scroll annotations begin as hand-labelled strokes and are refined through iterative pseudo-labeling; there is no exposed infrared ground truth for these scroll surfaces. Describe these as **public reference annotations**, not independently authenticated ink truth.

The complete public inventories contain 4,650 chunk objects plus `.zarray` per role. Their object dates identify the **18 August 2026** corrected upload. The preceding Russo audit records 921 added w029 supervision pixels and no added w029 validation pixels in that correction. This is useful historical evidence, but current byte-level masks/ETags remain the authoritative inputs for this study. Hugging Face buckets are mutable; object size/ETag pins and retained SHA-256 protect the observed version. An Xet hash/ETag is not asserted to be the plaintext SHA-256.

Current official loader semantics are verified at immutable Villa **e0bbb8b40a2db58b1d71864f286eb85717e59e64**:

- [dataset.py lines 292–303](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/data/dataset.py#L292): non-validation flat patches remove validation voxels from supervision.
- [normalization.py lines 62–80](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/data/normalization.py#L62): copy supervision and set it to zero wherever validation > 0.
- [patch discovery lines 123–136](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/data/patch_finding_default.py#L123): separate held-out patches use validation as their supervision override.
- [flat loss preparation lines 169–176](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/training/train.py#L169): labels are `amax > 0`; supervision is depth `amax`; nonpositive supervision is ignored unless full supervision is forced. Current config defaults that override to false.

Thus raw supervision membership is **not** sufficient proof of actual labeled training exposure. Primary validation must not blindly subtract raw supervision. Retain labels, supervision and validation independently; report their overlap. A derived `current-loader-label-mask = public_supervision>0 & ~(validation>0)` describes the current loader applied to these public manual masks, not all four models' historical exposure.

Both retained baseline embedded configs and the actual Russo control/ex016 embedded configs omit `force_full_supervision` and `dynamic_label`. Exact historical trainer revisions remain unestablished. Russo ex016 uses pseudo-label replacement on w029; its builder excludes the old validation region, but its full pseudo supervision/teacher history was not acquired here. Public manual supervision is not that treatment's full training mask. Neighboring patches can also contain validation CT as input context while masking its labels. Physical alias, complete sampling-support and teacher independence remain unknown. The primary result is a **mask-defined developmental annotation-agreement diagnostic**, not an untouched physical test set. Checkpoint filenames indicate 75k, while the separate actual checkpoint audit observes stored step **74,999**; that contract belongs to the model audit.

## Selection, observed counts and staged budgets

Metadata-only ranking was fixed before decoding labels: descending count of potential validation chunks, descending potential positive-label chunks, then ascending source Y/X. Object sizes unequal to the common 78-byte baseline were **ranking proxies only**. Sixteen chunk-aligned 256² candidates used 31 unique XY chunks across three roles: at most **93 objects**, **10,464 expected compressed payload bytes**. The gate selects the first with at least **256 positive and 256 negative validation pixels** in fixed local **[64,192)²**, before scan bytes or predictions; it has no threshold relaxation.

Root executed the independently reviewed [selection plan v2](label-selection-plan-v2.json) and [selector v2](select_w029_labels_v2.py). The first candidate passed:

| Quantity | Actual selected labels |
| --- | ---: |
| Level-2 source XYWH | **[6912, 3200, 256, 256]** |
| Fixed interior validation | **16,384** |
| Interior positive / negative | **7,006 / 9,378** |
| Whole validation | **51,580** |
| Whole validation positive / negative | **27,818 / 23,762** |
| Public supervision / raw S∩V | **0 / 0** |
| Ink / validation unique values | **{0,1} / {0,1}** |
| Supervision unique values | **{0}**; no positive supervision encoding is observed in this crop |

Selection record [selection-v2/selection.json](selection-v2/selection.json), SHA **`9e6f8b6857d354575e3cdb125cf811171ec85f3ecdecd219788ffc23e9f05bbd`**, was published last after input guards. Root reports **12 compressed objects, 24 HTTP requests, 13,414 received body bytes including redirects**. V2 disables automatic redirects, permits at most six HTTP requests per object, and enforces shared **558-request / 512 KiB body** caps. Signed redirect queries, Location headers and transport-error strings are not logged. Frozen v1 is retained unexecuted because its automatic redirect-body accounting was inadequate.

The separately reviewed [scan plan v2](scan-assembly-plan-v2.json) and [assembler v2](assemble_w029_surface_v2.py) bind the selected bbox and source identity. Exactly four uncompressed source chunks, ZYX **[0,25,54], [0,25,55], [0,26,54], [0,26,55]**, total **7,143,424 bytes**. Fresh source metadata before and after adds **7,346 expected body bytes**. The shared source cap is **64 MiB / 48 HTTP requests**, including redirect/error bodies; no automatic retries or missing-object fill. Parent owns execution. Frozen files are unchanged during execution; source provenance is published last.

The driver-facing provenance supplies hashed `surface`, `inklabels`, `supervision_mask`, `validation_mask`, `acquisition_observed_mask`, and `source_observed_nonzero_mask` arrays. **Acquisition-observed** means actual successful source-object delivery, including observed zero values. **Nonzero occupancy** is exactly `np.any(surface[2:19] != 0, axis=0)`, used for diagnostic stratification only. No geometry-validity mask is supplied. Primary support is validation membership ∩ acquired-observed ∩ fixed interior, subsequently intersected with common finite/evaluated model support. Both class-count gates must still pass there. Geometry, physical sampling independence and authentic ink remain separate evidence questions.

Metadata/source research received **4,440,477 observed final-response bytes across 32 recorded responses**. The initial metadata helper enforced a 5 MiB final-body cap but followed redirects automatically; **redirect-body overhead and actual total HTTP transactions were not measured**. Do not label that number total wire traffic. This defect is explicitly retained; v2 selection/source transport corrects it. There were no further network reads by this audit after the metadata review. Headers/TLS bytes are outside the response-body budgets.

## Terms and claim limits

Modern public CT defaults to CC-BY-NC 4.0 under the [retained official data terms](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/docs/02_data.md). Curated annotation documentation links [legacy source terms](https://dl.ash2txt.org/LICENSE.txt), including redistribution and public hidden-text/related-code restrictions. Their exact reach across current annotation versions remains unresolved: preserve private research artifacts and resolve publication/release permission before redistribution. Code licenses do not relicense the annotations. No terms acceptance, publication, submission, provider action or new account operation was performed by this audit.

Using another scroll's public high-resolution reference does not itself invoke Grand's ban on data derived from higher-resolution scans **of the submitted target volume**. A later research decision to use higher-resolution PHerc0800-derived data would need that eligibility distinction recorded; scientific usefulness and award eligibility should not be conflated. This small pooled w029 comparison does not establish native 8.64 µm PHerc0800 transfer, novelty, legibility or any prize claim. Independent pooling/hash verification will be retained additively after source acquisition completes.
