# PHerc1667 native reference source preflight

Audit date: October 8, 2026 (UTC). Experiment: `pherc1667-native-reference-v1`. Source and metadata audit only; experiment remains blocked and not ready for GPU execution.

**No verified public source identity for the paper aliases `l_2` or `l_5` was found in the inspected sources.** Do not substitute `w013`, another current PHerc1667 surface, or an arbitrary 62-plane window. Metadata confirms structurally bounded public chunk access, but it does not establish the exact ablation inputs, aligned labels, or physical depth convention.

## Paper and checkpoint identities

[Supplementary Table 2](https://arxiv.org/html/2606.29085v1#Sx13.T2) names `l_2 (PHerc.fragment)`, `labels/l_2_inklabels5.png`, and inference-only `l_5`; it does not give public segment IDs, scan URLs, or an alias crosswalk. The pseudo-labeling ablation is **[Extended Data Figure 7](https://arxiv.org/html/2606.29085v1#Sx12.F7)**; the plan's Figure 6 citation needs correction by its owner.

The pinned model cards for [iteration 0](https://huggingface.co/scrollprize/PHerc.1667-iteration-0/raw/06c306449b39df42c745608ceade243498d24243/README.md) and [iteration 5](https://huggingface.co/scrollprize/PHerc.1667-iteration-5/raw/6288ba60115ce827018926bcb0ea8e5696f33d3b/README.md), and their configs, specify 62-plane, 256 × 256 input. Iteration 5 uses `l_2_inklabels5.png`; iteration 0 describes cross-segment training and `reverse_layers=True` for `l_2` inference. They do not map either alias to a public physical segment/version. Model-card MIT licensing does not by itself establish dataset or annotation licensing.

The [curated PHerc1667 ink directory](https://huggingface.co/api/buckets/scrollprize/datasets/tree/ink/1667?recursive=false&limit=100) contains `w013`, `w018`, `w023`, `w028`, `w029`, and `w031` directories without an alias crosswalk. The [pinned original code metadata](https://raw.githubusercontent.com/ScrollPrize/villa/e583fb67468f483fadd73d52e06f0ab0fe5ba813/ink-detection/metadata.json) also does not resolve `l_2`/`l_5`; its `l2_0_crop` name belongs to PHerc0139 and must not be treated as `l_2`. These are findings about the inspected sources, not proof that a crosswalk exists nowhere.

## Verified current public metadata

The [official metadata index](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json) was revalidated with HEAD and conditional GET: ETag `"b4788ae14c64e13e57346c07c3f0b8a5"`, last modified October 5, 2026 at 15:58:43 GMT, conditional response 304. The original archived gzip snapshot remains unchanged (SHA-256 `7ec8dcaa3000b9716724c26a1991ee416491d8a6b3615238ca90ae11540abdd3`). All 19 surface `.zattrs` and native `0/.zarray` records, plus the target volume metadata, returned HTTP 200 and are retained with exact URLs, hashes, ETags, and lengths in the immutable metadata manifest.

All 19 released native surfaces target volume `20251217075048`, scan `20251204164204`, at **2.399 µm**. They contain **109 planes**, not the models' 62-plane input. Their native scale is `[2.399, 2.399, 2.399]` µm in Z/Y/X, `slice_step=1.0`, `source_group=0`, and uint8 values. Render records date to June 17–19, 2026. Recorded settings include `flip-normals=true`, scale 1, rotation 0, flip −1, and zero crop parameters. Renderer commits are `ae2ecce9dded502b0cc24127e9fceabd04e1257d` for `w013` and `650bd56663efb7225fdba73fe267ba86b8e29368` for the other 18.

| Public segment ID | Public name | Native H × W | Compressor |
| --- | --- | --- | --- |
| 20251212185248 | w029 | 38000 × 31320 | None |
| 20260119120000 | w036 | 39900 × 33160 | None |
| 20260205070000 | w041 | 34200 × 30540 | None |
| 20251208130119 | w028 | 37420 × 30340 | None |
| 20251223230000 | w031 | 37480 × 32180 | None |
| 20260116230000 | w035 | 36680 × 32560 | None |
| 20251206103305 | w012 | 39380 × 15700 | None |
| 20240304144031 | w018 | 42380 × 98100 | None |
| 20260123230000 | w037 | 34260 × 31780 | None |
| 20260130150000 | w039 | 36100 × 33280 | None |
| 20251220020000 | w030 | 37800 × 31720 | None |
| 20260115160000 | w034 | 35900 × 32440 | None |
| 20260110010000 | w033 | 36100 × 31580 | None |
| 20260128140000 | w038 | 35180 × 32900 | None |
| 20240304141531 | w013 | 41600 × 79600 | Blosc |
| 20260203210000 | w040 | 35120 × 31520 | None |
| 20260105050000 | w032 | 38060 × 32820 | None |
| 20260108140509 | w011 | 39500 × 14720 | None |
| 20240304161941 | w023 | 41860 × 102360 | None |

A further merged `w011` segment in the index lacks a native surface-volume record and is outside this 19-surface inventory. Exact long IDs and URLs are retained in the manifest rather than reconstructed from aliases.

For the previously used known-ink source `w013`, the verified current native array is **`[109, 41600, 79600]`**, from [20240304141531-w013_20240304141531_flatboi metadata](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1667/segments/20240304141531-w013_20240304141531_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/.zattrs). Its spatial level 2 has 9.596 µm Y/X spacing while preserving the native 2.399 µm depth spacing. The old control then pooled real depth planes to 21 means. That control is neither the native paper input nor evidence that `w013` equals `l_2` or `l_5`.

The native CT source metadata identifies `mask_layers_zarr_export`, with float window −0.03 to 0.21 and uint16 window 7,396 to 45,012. It describes a masked/windowed uint8 source volume, shape `[37076, 15229, 15229]`, chunks `[128, 128, 128]`, and no compressor. This source provenance does not establish numerical equivalence to the original paper TIFF intensities. Clipping the current uint8 data to the model-card range 0–200 cannot repair an unverified source version or quantization mapping.

The exact renderer sampling implementation at these two current commits was not inspected here. Stored axes, step, and normal-flip settings are verified; the signed physical depth order, center-offset formula, and exact 62 source indices for each paper alias remain unresolved. No transform, slice subset, or orientation was fitted or guessed.

## Bounded acquisition feasibility

Every native surface array has chunks **`[109, 128, 128]`**, dtype `|u1`, C order, and fill value 0. Each decoded full chunk is **1,785,856 bytes (1.703125 MiB)**. A future chunk-aligned 1024 × 1024 ROI requires **64 XY chunks**: **114,294,784 bytes (109 MiB)** for all 109 planes. The proposed three `l_5` contexts plus one `l_2` sanity region would therefore require **256 chunks and 436 MiB decoded**, before annotation/geometry assets. No such regions were selected or acquired.

For the 18 uncompressed arrays, 109 MiB is also the full extant chunk-body payload per crop. Their native chunk keys follow `<surface-base>/0/0/<chunk_y>/<chunk_x>`. The `w013` array uses Blosc level 3 and dotted keys `<surface-base>/0/0.<chunk_y>.<chunk_x>`; its exact compressed wire cost is not established without chunk responses. Chunk HTTP success/existence and HTTP byte-range support were not tested. Missing chunks with fill value 0 must be distinguished from observed, geometrically valid material in any future acquisition. No partial-depth range route is verified, and the required 62-plane window is unknown.

Mesh coordinates, mesh metadata, full geometry masks, and original aligned `l_2` annotation/supervision masks were not acquired. The index records transformed TIFF-XYZ meshes, but this bounded pass fetched surface and volume metadata only. Consequently, the plan's geometry-based crop conditions and annotation alignment are not yet executable. Resolving a public source/version crosswalk and the original depth/label contract comes before crop selection or scan acquisition.

## Accounting and retained evidence

This preflight made **54 read-only requests**: 42 metadata requests with exactly **73,110 response-body bytes**, plus 12 alias-source audit requests charged conservatively at **at most 196,087 response-body bytes**. Aggregate response-body usage is **at most 269,197 bytes**, under the authorized 20 MiB cap. One early alias-source response did not retain an exact count and was charged its 100,001-byte read cap. Counts exclude HTTP/TLS overhead. No weights, scan chunks, TIFF images, predictions, training, provider actions, or spending occurred.

Metadata manifest: `/workspace/herculaneum-lab/data/native-reference-preflight-20261008/source-3edb00b5544a20658ab6d98069fbc841d773cd4df74768477371f77390a4e607-metadata-manifest.json` (SHA-256 `3edb00b5544a20658ab6d98069fbc841d773cd4df74768477371f77390a4e607`). Final source/depth/acquisition manifest: `/workspace/herculaneum-lab/data/native-reference-preflight-20261008/source-7af9c2b2d252dae2896f01dd3d50df6de8bda110ba1745916170ade22a9877ef-preflight-manifest.json` (SHA-256 `7af9c2b2d252dae2896f01dd3d50df6de8bda110ba1745916170ade22a9877ef`). The final manifest pins the local plan, paper, model cards/configs, prior control audit, source URLs, observed array layouts, unresolved identities, and network accounting. Both manifests are content-addressed and read-only. Original tracked files and scientific input/provenance files were not changed.
