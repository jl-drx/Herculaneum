# Fragment label reader design — route not yet frozen

This is preparation for a reference contract, not an acquisition or evaluation plan. Initial metadata preparation read only retained metadata, inventories and installed source code. Root subsequently authorized the local decoder and tiny synthetic fixtures. No candidate arrays were decoded and no network requests or model calls were made. The accompanying `reader-source-receipts.json` records the initial metadata-only phase's local sources, versions and inventory identities. Root must freeze the scientific source-to-label contract, selection rule and budgets before any candidate acquisition.

## Recommended route for PHerc0009B

The current complete S3 inventory (`IsTruncated=false`) lists one level-2 encoded shard per role:

| Role | Key suffix | Encoded bytes |
| --- | --- | ---: |
| Ink labels | `inklabels.zarr/2/c/0/0` | 17,784 |
| Supervision | `supervision.zarr/2/c/0/0` | 7,830 |
| Validation | `validation.zarr/2/c/0/0` | 6,538 |
| Total | Three complete objects | 32,152 |

Three bounded whole-object HTTP 200 reads are preferable here to separate index and payload range requests. Preserve each compressed object privately with SHA256, length, URL, inventory ETag, response headers and a shared request/body ledger. Decode only the selected inner chunks. A 35.75 MiB decoded outer shard does not imply a large network object. Do not let a high-level remote Zarr or fsspec reader issue untracked HEAD, range, listing, retry or full-object requests.

The retained eighth `bounded_http.py` already pins expected length/ETag, requires identity transport encoding and HTTP 200, records SHA256, restricts public hosts and bounds requests/body bytes. It needs a narrow additive version for this stage rather than an edit to frozen code: zero automatic retries, preferably no redirects for these exact S3 URLs, and incremental body accounting that preserves partial bytes when a read raises. Its current single `response.read(expected)` can lose a partial body from the ledger if an exception occurs before assignment. Log failures without response text or signed query strings. Expected ETag is an object identity check; do not assume an S3 ETag is a content SHA256.

Success payload is 32,152 bytes. Root must separately freeze limits for metadata, failures and request attempts; this document does not authorize a budget or download. No mapping-validity object is present anywhere in the current complete release prefix. Catalog `validity_published=true` is contradicted by that inventory and is not a substitute for an acquired mask.

## Exact supported codec contract

PHerc0009B ink/S/V level-2 metadata is identical: Zarr v3, uint8 `[6340, 5130]`, axes YX, fill value 0, outer regular shard shape `[6656, 5632]`, default key separator `/`, one `c/0/0`. Inner chunks are `[512, 512]`; the inner codec chain is bytes then Blosc zstd, clevel 3, bitshuffle, typesize 1, blocksize 0. The shard index uses little-endian bytes followed by CRC32C and is stored at the end.

There are `13 × 11 = 143` index entries. Each is two uint64 values, so the trailing index is `143 × 16 + 4 = 2,292` bytes. Parse C-order `(13, 11, 2)` entries only after validating the Castagnoli CRC32C checksum; CRC32 is a different algorithm. Use explicit little-endian uint64/uint32 handling and confirm parity with the installed Zarr codec in a local fixture. CRC protects the index here; the metadata does not declare a separate payload CRC.

Installed Zarr `sharding.py` explicitly treats index offsets as absolute offsets from the beginning of the final encoded shard. For an end index, payload byte zero is shard byte zero. Do not add the index length or treat offsets as relative to a detached payload section. A start-index fixture should verify that start-index offsets are also absolute; the production adapter may reject start-index metadata because it is outside the frozen production contract.

Before decoding any inner payload, validate every index entry, using Python integers for addition:

- Missing entries have both offset and length equal to UINT64_MAX; reject a half-missing pair.
- Present lengths must be positive, and `[offset, offset+length)` must lie wholly before the trailing index.
- Reject overflow, truncated entries and overlapping present payload intervals. Exact aliases should also be rejected unless independently justified and explicitly added to the contract.
- Reject any unsupported shape, dtype, codec, endian, fill value or index location rather than silently adapting.

Each decoded inner chunk is exactly 262,144 bytes, including an edge chunk's stored padding. Check the Blosc header's declared uncompressed size, compressed length and typesize before decompression. Use a bounded preallocated output and verify exact resulting length. A per-payload bound of 262,160 bytes (raw size plus the installed Blosc maximum overhead of 16) is a proposed guard to verify in the synthetic fixture. Decode one chunk at a time, then copy only the requested, in-array intersection; do not expose padding outside `[6340, 5130]`.

Installed packages already support the codec chain: Zarr 3.4.0, numcodecs 0.17.0 and google-crc32c 1.9.0. No installation is needed. `numcodecs.Blosc.decode(..., out=...)` is the public decompression interface. Header helpers `_cbuffer_sizes` and `_cbuffer_metainfo` exist but are private; prefer an explicitly checked header parser or pin and fixture those helpers if used. Installed source digests and all inspected package versions are in the source receipt.

For comparison, retained PHerc0500P2 level-2 label metadata is `[6190, 3470]` with outer `[6656, 3584]`, the same inner codec chain, and a `13 × 7` index of 1,460 bytes. Its present metadata is sufficient to describe the decoder, not to authorize a second specimen's acquisition.

## Fill, participation and provenance

An absent inner chunk in a successfully acquired, CRC-valid shard means Zarr's declared fill value. It is different from a failed HTTP request or missing object. Preserve an `encoded_chunk_present`/`canonical_fill` diagnostic when assembling label ROIs. Root must explicitly approve canonical fill semantics for each role. A filled ink-zero can be a negative only where an independently supplied participation mask permits it; it must never turn an unlabelled pixel into background supervision.

Do not synthesize measured mapping-validity 255, geometry validity or annotation certainty from supervision, nonzero source profiles, fill zeros or successful transport. Mapping validity remains unavailable. S and V remain separate raw arrays with separately verified observed encodings; role membership is not established until their data is decoded under a frozen label-only plan. Source occupancy and successful CT acquisition are different diagnostics from mapping validity.

An eventual ROI receipt should bind metadata/inventory/source-code SHA256s, full shard URL/size/ETag/SHA256, index length/checksum, selected inner coordinates and absolute payload ranges, compressed and decoded lengths/hashes, output shape/dtype/hash, requested YX bbox, edge clipping and canonical-fill coverage. Record all request bodies, including failures, against one cap; never retry automatically. Recheck pinned inputs before and after assembly.

## Coordinate and depth policy: unresolved

Label root attrs declare canvas YX `[25360, 20520]`, level-2 scale `[9.604, 9.604]` µm and `canvas_offset_yx=[-38.351, -44.944]`. The separate [source contract audit](../surface-contract/CONTRACT.md) traces official transfer code at `3ea17f54a9b3d5fd1aaf73e1d2c8386dbaa9f30e`: the offset corrects the full-resolution source label sampler during transfer, while output indices belong to the target canvas. Properly transferred output therefore intends zero additional canvas-offset shift. Adding the recorded offset to the output again would repeat the source correction. Exact historical execution remains unverified: public attrs have no applied flag/OME translation and name private reports plus an absent `provenance.json`; the publisher revision is dirty and the inspected public helper differs from this release's 2D format. Matching shapes/scales do not independently prove registration. Do not apply, divide, round or fit an offset in this reader. The CT mean-pyramid versus categorical-label decimation also has a 0.375-coarse-pixel kernel-centroid phase; no correction has been selected. Root's scientific contract owns those decisions.

The surface is uint8 Zarr v2 `[109, 6340, 5130]`, uncompressed chunks `[109, 128, 128]`, with level-2 spacing ZYX `[2.401, 9.604, 9.604]` µm and explicit source slice step 1. The archived renderer records 109 slices, flipped normals and a geometry source. That can establish a nominal rendered surface center after auditing the renderer, but does not establish the 2D label's depth center/window or model-ready pooling. Do not reuse earlier `[13,97)` pooling or `2:19` model layers merely because another source had 109 slices. PHerc0500P2 instead has 118 slices at Z spacing 2.215 µm, which reinforces the need for an explicit source-specific contract.

Without public mapping validity, offset application and label depth support, the reader can at most prepare a conditional supplied-label agreement diagnostic. It cannot establish measured-only, optical, human-only or historically independent accuracy. Both candidate specimens and their aliases remain outside the current adaptation/calibration until root assigns a role.

## Optional range route if an object is larger

The current frozen readers do not support a Range header or HTTP 206 validation. If root later chooses larger shards, implement a separate strict adapter rather than passing range requests to the old HTTP 200 reader. First pin object size and strong ETag/version using a frozen inventory or bounded HEAD. Obtain the 2,292-byte index tail (or the source-specific computed size), then selected payload ranges with `If-Match` and identity encoding. Require 206 with exact Content-Range start/end/total, exact Content-Length and matching object identity. Reject HTTP 200 fallback, changed ETag/412, redirects outside the frozen policy, truncation and excess bytes. Count every response incrementally, including failures. Use half-open internal ranges and convert to inclusive HTTP endpoints deliberately. Apply the same CRC, absolute-offset, overlap and decompression checks as the whole-object route.

## Meaningful fixtures before an acquisition freeze

Root authorized synthetic-only checks. The first installed-Zarr fixture passed 41 parity and rejection cases; its retained receipt binds unchanged decoder source. Its tiny 4×4 inner chunks use Blosc's memcpy fallback. An additive 64×64-inner fixture passed eight cases with genuine zstd/bitshuffle compressed payloads, including exact local-Zarr ROI parity and pre-decompression corruption/header rejection. The independent reviewer separately passed 44 checks, including the Castagnoli known vector, actual compression and reversed physical payload order. These are mechanical checks; no candidate arrays were read. The following lists the fixture coverage and optional future transport/transform tests:

1. A tiny asymmetric, multichunk Zarr v3 array written by installed Zarr with end-index bytes/CRC32C and Blosc zstd/bitshuffle. Compare selective decoding and partial edge ROI assembly exactly with standard local Zarr. Include an omitted fill chunk and an explicitly encoded all-zero chunk.
2. A start-index fixture showing absolute-offset semantics, even if production rejects that metadata. Reject corrupted CRC/index bytes, truncated indexes, half-missing sentinels, payload/index overlap, overlapping entries and out-of-bounds/overflow ranges.
3. Blosc header and payload rejection fixtures: wrong uncompressed size/typesize/compressed size, truncation and corruption. Confirm bounded `out` decoding and padding clipping.
4. Mock HTTP only: exact successful body, ignored-range 200 for optional range mode, wrong Content-Range/length/ETag, short body and an exception after a partial body. Verify all consumed bytes are charged and no retry or fill occurs.
5. A coordinate transform fixture only after the actual applied-offset and units policy is resolved, exercising axis order, factor-of-four scaling, fractional offsets and inverse mapping without inventing the production transform.

The proposed narrow interfaces are `validate_metadata(record)`, `validate_shard_index(encoded_bytes, metadata)`, `decode_role_roi(encoded_bytes, metadata, bbox_yx)` and a bounded whole-object fetch wrapper. The decoder should be pure/local and callable from fixtures without any network or model imports. Future root plans supply candidate order, annotation gates, maximum decoded chunks/bytes, mask semantics and exact depth/transform choices; the reader must not choose attractive regions or inspect predictions.
