# w040 teacher and target provenance admission

11 October 2026. This read-only investigation fetched source text, metadata and **6,483 TIFF metadata bytes**. It fetched or decoded **no teacher, label or supervision pixel data**. All TIFF metadata byte ranges were checked to be disjoint from every compressed image-tile range. No model or training was run. Evidence is retained beside this report.

**Admit a bounded teacher-only correspondence-sensitivity measurement using the released teacher's declared fine full-raster coordinates. Do not claim historical training-target reproduction, manual-label accuracy, physical ink accuracy, or resolution of the previous CT correspondence failures.** The origin evidence is stronger than matching dimensions alone: declared input lineage, published v42 source coordinates and TIFF metadata all agree. The object is not accompanied by a historical content checksum or immutable image digest.

## Exact teacher and declared coordinate contract

Public segment: `20250831000000-w040_2025083102`. Teacher filename:

`PHerc0139-20250831000000-2.399um-0.22m-78keV-volume-20260102150214-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif`

The complete HTTPS URL is in `TEACHER_TIFF_HEADER.json`. It lies in the public segment's `ink-detection/` directory. This exact filename appears in Nieuwlaar's `native_dense_report.json` at pinned commit `217e7521670db49cde97872ea6a8bdfb80c449f1`. The report records w040's native canvas `[28,6400,7980]`, own-calibrated threshold **48**, and historical manual override count **745,601 ink pixels**. `teacher_sources.tsv` names other windings and does not itself include w040; the report plus catalog provide the w040 linkage.

Retained catalog SHA256 `2a460d1f57ca0d3535ce2a8a8ab8b80fe4141f143e8de486394a150bc93f094d`, extracted into `CATALOG_TARGET_EXTRACT.json`, says this prediction was generated 19 June 2026 from fine78 surface volume **20260102150214**, using model **20260417190342**, `new_canon_autoresearch_recipe`, source layers **[23,85)**, tile 256 / stride 128, `force-reverse=false`, and no scale bar. Catalog image tags are `ink-detection:v42` and `:v42-cpu`.

The public annotated tag `docker/ink-detection/v42` resolves through tag object `6b91f905080b09e06e6fd9c8980789886090d8cf` to source commit **d4615b134ce1b62368bed1029ed47e5569ca03fb**, dated 22 April 2026. The pinned build script shows the publication mechanism names those container tags and then creates that Git tag. Source provenance is retained as `sources/villa-v42-{ref,tag}.json` and `sources/v42-*.py`.

Relevant source operations:

- `v42-inference.py` `LayersSource`, `preprocess_layers`, and `create_inference_dataloader` retain input H/W. Reading `(y1:y2,x1:x2)` and writing the prediction use the same XY indices. There is no global crop, affine warp, flip, added margin or origin translation in this path.
- `entrypoint.py` sets model input size equal to tile size; it supplies the declared Z interval. XY prediction is interpolated back to the same tile dimensions and accumulated at the source ROI coordinates.
- `v42-processing.py` `reduce_partitions` uses the existing prediction canvas without resizing. `write_tiled_tiff` writes that full canvas. The teacher's header description `{"shape": [24880, 31040], "software": "optimized_inference"}` matches this source contract.
- TIFF dimensions are **height 24,880, width 31,040**, matching the retained fine level0 render, and its orientation is default top-left. Native canvas is **6,400 × 7,980**.

Thus the teacher is represented as a 2D map on the original fine render XY canvas. It is not an independently annotated depth plane: predictions use 62 fine layers and the returned 2D target has no ink-surface Z coordinate. Any transfer through 3D CT queries must declare its surface/depth convention. A label's apparent stability under XY alternatives cannot certify the physical ink face.

The output contract is a sigmoid/weighted-average prediction clipped to [0,1], multiplied by 255, and cast to uint8. This supports treating the values as probability-style pseudo-targets, not calibrated probabilities or authenticated ink labels. The source zeroes pixels invalid over its input Z window; quantization can also produce zero. Do not equate every zero with independently verified absence of ink.

## Historical identity and manual overrides

The TIFF HEAD currently identifies **66,742,237 bytes**, multipart ETag `"5bb72bf41bb6feacb28a626e6730ec7a-8"`. That ETag is a concurrency/identity validator for these requests, not a whole-object SHA256. The sources provide no historical teacher-byte checksum. Container tags are published source evidence, not immutable image-digest attestation. Atlas metadata records a dirty workflow checkout. These limitations preclude claiming a bit-for-bit historical teacher or full model-ancestry reconstruction.

Pinned `native_common.py` loads the teacher, optionally shifts uint16 by 8 bits, then applies `cv2.resize(..., INTER_AREA)` to the native H/W. `build_native_dense_labels.py` binarizes w040 at 48, uses `native_valid & (resized_teacher > 0)` for pseudo supervision, and overwrites ink targets wherever the sparse manual supervision mask is set. Both label families store content at z14 in a 28-plane container. That storage plane is not proof of physical ink position.

Current public HF native metadata fetched without data chunks:

- `ink_9um/labels/native9-scrollprizeorg-21slices/w040/w040_inklabels.zarr`
- `ink_9um/labels/native9-scrollprizeorg-21slices/w040/w040_supervision_mask.zarr`
- Both use Zarr v2, `[28,6400,7980]`, `[28,128,128]` chunks, uint8, Blosc zstd clevel3 bitshuffle. The default missing chunk is zero.
- Their `.zattrs` name unavailable `/scratch/elian/mask-align/native-rerun/w040/...` images and transfer reports, plus a `rerun_provenance` block. This is declared native-frame supervision, not a new independent physical registration certificate.

The September S3 republication is under `ink-labels/9.362um-volume-20250728140407/20260918/{inklabels,supervision}.zarr/`. Its metadata says re-encoding from the public HF arrays preserves canvas, geometry and pixel values, removes Z padding, and uses original source plane14. The catalog counts **745,657** ink pixels and **1,629,734** supervision pixels. **745,657 differs from the historical builder's 745,601**, so live manual arrays cannot be silently substituted as exact historical overrides. Current native label metadata has no independent per-pixel mapping-validity array.

The fine September manual package is a separate transferred label release. Its offset `(dy,dx)=(30.2013538561,-50.2388588815)` describes that annotation-transfer lineage. It is not an offset to apply to the earlier canonical teacher TIFF. Do not mix these coordinate contracts.

A teacher-only sensitivity test may report how pseudo-targets change before manual overrides. It may not report how many historical final training labels changed without identifying the exact historical override bytes. If current manual support is separately measured, label it current-version support and keep its contribution separate.

## Bounded reader contract for the next frozen step

No pixels have been read here. `TEACHER_TIFF_HEADER.json` preserves the complete tag values, including `TileOffsets` tag324 and `TileByteCounts` tag325, with raw metadata ranges and their SHA256 values retained separately. These suffice to fetch only tiles intersecting the prospectively frozen sampling footprint.

- Single image, no next IFD; unsigned 8-bit, one sample per pixel, min-is-black, little-endian classic TIFF.
- Tile **1024 × 1024**, row-major grid **25 × 31**, 775 tile entries. Compression tag **32946 Deflate**, no predictor tag (default1), no palette or extra sample. Tile index is `tile_y * 31 + tile_x`.
- Freeze the required fine-coordinate bounds, interpolation halo, tile index set and compressed-byte budget before fetching. Include all geometric alternatives in that footprint. Do not alter support after viewing labels.
- Request each exact interval `[offset, offset+bytecount)` using `Range: bytes=offset-(offset+bytecount-1)` and `If-Match` with the retained ETag. Require HTTP206, exact `Content-Range` and original total length, matching ETag, and exact response length. Fail if the server returns an unrestricted HTTP200, an unexpected version or inconsistent metadata.
- Save each compressed tile once with its SHA256, URL, range, ETag and retrieval timestamp. Decode with the standard TIFF Deflate/zlib codec; enforce the full tile's expected **1,048,576 output bytes**, end-of-stream and no trailing/unconsumed data. A bounded decompressor should reject output exceeding this size before allocating more. Reject unexpected compression/predictor/sample layout.
- Tile coordinates include edge padding, but the declared image bounds remain 24,880 × 31,040. Discard padded cells outside that canvas. Assemble only the frozen bounded window with an explicit canvas-origin record; hash the decoded crop and preserve its dtype/shape.
- Distinguish the newly pinned current-object subset from a whole-object historical checksum. These byte subsets plus metadata are enough to replay the bounded audit, not to reproduce the whole teacher image.

`ADMISSION_SUMMARY.json` records checks that all TIFF image ranges are within the object and disjoint, and that all ranges fetched during this metadata audit are outside image data. Recompute these checks before adding tile responses to a new ledger.

No stable pseudo-target result should relax the earlier CT gate retrospectively. This packet establishes a defensible declared coordinate contract for measuring supervision sensitivity, not a new training admission or ink reading.
