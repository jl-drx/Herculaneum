# Native-frame reference coverage intake

Frozen 11 October 2026 before label or supervision pixels. Scope: current public native annotation support and storage semantics for PHerc0139 w035, w039 and w044. This is a supporting data-readiness study, not a training experiment, independent physical ink truth, or a validation of the unavailable historical annotation-transfer workflow.

## Source and acquisition

Use only the six current S3 Zarr v3 level0 arrays in `SOURCE_PLAN.json`: one supervision and one inklabels array for each winding. Each has one outer shard at `c/0/0`, uint8, 512-square inner chunks, Blosc zstd/bitshuffle and a CRC32C-protected index at the end. Array canvas matches the native CT metadata. These arrays declare re-encoding of native-frame source plane14; do not apply the w040 fine-teacher coordinate correction. Plane14 is storage layout, not proof of physical ink depth.

Acquire all three complete supervision shards first, in winding order w035,w039,w044. Do not acquire label shards until the deterministic mask-only eligible-core list has been saved and hashed. Then acquire all three complete inklabel shards in the same winding order. No CT, alternate labels, image interpretation, source search, model, training or GPU is included.

Exact URL, HEAD length, ETag and metadata hashes are pinned in `SOURCE_PLAN.json`. Use HTTPS GET with If-Match and no redirect/retry. Require HTTP200, exact body/Content-Length and same ETag; reject errors or changed versions. Enforce a cumulative4MiB network-body cap across these six objects, including any unsuccessful read. Current expected total65,648bytes. A changed object is a stop, not permission to repin. Save exact source bytes, SHA256, response validators and retrieval times. No complete historical-source identity claim follows from pinning these current objects.

## Decoder and integrity checks

Check every declared array/shard/chunk dimension, dtype, fill value, codec and index layout against the retained metadata. Check index CRC32C, exact entry count and end-index placement. An absent inner chunk must use the specified paired uint64 sentinel and decodes to fill zero. Every present chunk must have positive compressed length, remain wholly before the index, and not overlap another present range. Decode each with official numcodecs Blosc, require exactly512×512 uint8 bytes, assemble in C-order, and crop only at declared array bounds. Independently decode through official Zarr and require byte-identical full arrays. Retain both decoded SHA256s and all index/codec checks.

Before any source pixel access, exercise asymmetric, edge-cropped and missing-chunk synthetic arrays, and rejection of corrupt CRC, out-of-bounds index and overlapping entries. A decoding or identity failure stops the intake. No missing shard becomes an all-zero annotation.

## Mask-only coverage grid, frozen before label acquisition

Supervision means nonzero uint8 value; record complete value histograms rather than assuming1 or255. Enumerate every128×128 output core whose top-left x,y is a nonnegative multiple of128, with stride128, in y-major then x-major order. Require its complete256×256 context `[x−64,x+192) × [y−64,y+192)` to lie inside the declared image. Consequently border cores without the64-pixel context are excluded; no padding or shifted replacement grid.

Retain **every** such core with at least1,024 nonzero supervision pixels inside the128-square core. Record all grid positions, counts, inclusion reasons and included context boxes. Save `MASK_COVERAGE.json`, per-winding full eligible-core CSVs and `MASK_STAGE_COMPLETE.json` with their hashes before any label GET. No ink values, class balance or visually attractive region may affect the eligible list. Grid cores do not overlap; contexts may overlap and are not independent samples.

For every winding record total supervised pixels and coverage; all candidate and eligible grid counts; eligible output area; eligible supervised area; and excluded support outside eligible cores. Compute the union of native128×128 CT chunks needed for all eligible256-square contexts at all28 source planes using the pinned native metadata. Save every unique `(z_chunk,y_chunk,x_chunk)` ID and exact uncompressed-byte estimate (`28×128×128` per current stored chunk). This is a prospective estimate; fetch no CT. Since contexts lie inside image, edge chunks may still encode padded native borders according to array chunk semantics; count full stored chunks.

## Labels and endpoint semantics

After the mask-stage receipt exists, acquire/decode the three fixed inklabel shards. Ink means nonzero stored value. Report full label histograms, total nonzero labels, ink inside supervision, labels outside supervision, supervised non-ink pixels, and all corresponding counts within every previously frozen eligible core. Save every core, including single-class or empty-ink ones; never replace/drop a core after labels. A label outside supervision is explicitly reported, not silently used as a negative or included as adjudicated training data.

Class summaries establish availability for a future frozen annotation-agreement endpoint. They are not ink-accuracy scores. The manual source may have its own alignment/transfer errors, and named windings have upstream model/training exposure. Do not call them pristine held-out tests. A later learning study must preserve winding/source ancestry and define train/development/evaluation roles independently. This intake does not assign model-selection roles, alter the prior correspondence gates, certify sealed-scroll transfer, or authorise training/GPU work.

Stop after full-array support/count semantics and the prospective native CT budget. Preserve negative or inadequate coverage outcomes without crop search. Package source shards, metadata, protocol, code, receipts, all eligible cores and checksum manifest for independent replay; no manually interpreted image panels are needed.
