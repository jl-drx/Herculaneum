# Phase27: native storage contract and synthetic reader qualification

The published native array uses **128×128×128 uint16 chunks**, not the provisional64cubed layout. Under the same retained-geometry probe assumptions, the source support is **70chunks /280MiB**, so the previous39MiB estimate does not apply. A strict local adapter passed20tests and exact reference-writer roundtrip checks on a129cubed synthetic volume. The64MiB source-chunk limit can be preserved with13contiguous point batches, but that schedule repeats130chunk uses. No candidate CT pixels were read and no GPU was rented.

## What was actually obtained

The prospectively selected public store was:

https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/volumes_zarr/53keV_3.24um_.zarr/

Four metadata requests were made, without retries or redirects. The first three200responses were retained and parsed: root .zattrs3018bytes, .zgroup24bytes and0/.zarray401bytes. Total returned body bytes were3443; conservative reserved body bytes102404. The fourth request,0/.zattrs, returned404 with chunked transfer and no Content-Length. The frozen fetcher rejected that framing before consuming its body. **Acquisition status remains STOP**, not successful completion. No retry, alternate store or chunk body was requested. The metadata-only fetcher's9offline fixture tests passed before acquisition.

A separately frozen READER_CONTRACT.md permits limited offline use of the three verified metadata files. It does not convert the fourth response into a validated absence, establish an atomic cross-object snapshot, or authenticate an entire array. Source object validators, lengths, hashes and failure details are retained in metadata-run-v1/request-ledger.json.

| Field | Published declaration |
|---|---|
| Format / dataset | Zarr2 / first multiscale dataset0 |
| Shape Z,Y,X |8855×2260×6300|
| Axes / scale |ZYX space axes; native scale[1,1,1]|
| Physical units / pitch |Not declared in these Zarr metadata objects|
| Chunks |128×128×128;4194304decoded bytes per physical chunk|
| Dtype / order |Little-endian uint16 /C|
| Compression |Blosc, Zstd, level3, shuffle0, blocksize0|
| Filters / separator / fill |None / slash /0|
| Sharding |No sharding codec in this v2 declaration|

The shape matches the retained native volume descriptor. This does not prove which energy/volume produced the historical PPM or render. The3.24um spacing and physical-unit ray interpretation remain explicit assumptions inherited from the previous geometry rehearsal. Dataset-level attributes and native chunk framing remain unverified.

## Offline qualification

No packages were installed and tracked repository source was unchanged. Python3.12.14,NumPy2.5.3,zarr3.4.0 andnumcodecs0.17.0 were used. An exploratory codec/writer API probe and a19test development pass preceded the frozen20test qualification run. The extra test checks full physical edge-chunk budgeting before object access. One frozen qualification attempt passed, with0failures/errors/skips, in1.361seconds. Peak processRSS was122720256bytes, about117MiB. **64MiB is the source-chunk budget per batch, not a process-RAM limit.**

The unchanged phase26 sampler receives data from local_zarr_reader.py. The adapter binds metadata and explicit synthetic object SHA256/length identities, checks physical decompression size before allocating a bounded output buffer, enforces the selected schema/codec, and crops padded storage chunks to their logical edge shape. Unknown objects stay missing; expected objects that disappear or change stop. Valid zero remains valid. Cache files publish only after decode and an atomic rename; incomplete .partial files are never accepted as completed data. The cache is a local single-writer prototype. The interrupted-write test injects a failure before rename; it is not a process-crash or lost-response test.

The reference writer produced a129×129×129 known linear phantom, with the actual128cubed chunk dimensions. All2146689voxels matched installed Zarr readback, and all8full/edge chunks matched known arrays through the custom reader. Sampling across the native chunk boundary, at the final voxel and at valid zero matched the analytic values within2.2737367544323206e-13; the out-of-volume point was invalid. Eight source objects totalling171878encoded bytes were read, then8verified cache hits were decoded for sampling. Synthetic compression sizes are **not** estimates for scroll scan transfer.

The20tests cover schema/axis/endian/order mismatches; changed metadata; missing, changed and truncated objects; freshly bound corrupt compressed payloads; impossible decompression headers; byte limits before source access; zero versus unknown; edges; cache reuse, altered cache, disk quota and interruption. The known source array and installed Zarr writer/readback supply the oracle. These tests do not prove compatibility with unseen native chunk headers, provide an authenticated live object manifest, or qualify HTTP/S3 transport.

## Geometry and storage accounting

All729retained PPMrecords were reused, with558mapped and171sentinels. The same three hypothetical probe depths[-3.24,0,+3.24]um yield2187ordered pointslots,1674geometrically valid and513invalid, with13392positive interpolation-corner contributions. No new geometry was downloaded.

| Quantity | Actual-layout dry plan |
|---|---:|
| Unique source chunks |70|
| Unique full decoded support |293601280bytes =280MiB|
| Whole-batch64MiB result |Rejected before loader access|
| Contiguous point batches |13|
| Chunks per batch |12batches of16; final batch8|
| Point slots dropped or replaced |0|
| Chunk decodes without cross-batch reuse |200|
| Repeated chunk decodes |130|
| Decoded work without cross-batch reuse |838860800bytes =800MiB|

The scalar second-pass checker independently enumerates corners and chunk keys in root-owned code, checks every range and support union, and confirms all saved identities. No independent-agent review is claimed. Preserving point order here is a conservative baseline, not an optimal schedule. The actual compressed/wire-byte cost and live-source latency are still unknown. Caching encoded objects can avoid repeated downloads but still repeats decompression; it is not equivalent to decoding once.

## Decision

Keep this adapter and corrected budget as an offline prototype. Next test **chunk-oriented accumulation**: load each chunk once, accumulate its weighted contributions into output slots, and preserve complete-support validity across chunk batches. Seventy chunks fit five batches of at most16chunks; that is a planning target, not a tested scheduler result. Test numerical equivalence, missing-chunk propagation, edge padding, interrupted accumulation and explicit coverage before any live candidate-data release. This addresses a measured inefficiency without paying for a GPU.

Historical source mapping, optical annotation/support, registration and model-specific exposure remain unresolved. The maintainer request remains user-intended, not confirmed sent; no outreach was performed. Frag6StageC stays closed, reserved candidates remain unscored, reused Paris contexts remain development and the existing model baseline is retained. No model forward, training, new candidate score, transcription, translation, upload, publication or prize submission occurred.

GPU spend remains£0 and the£10TOTAL first-pilot cap is unchanged. A paid launch still needs a useful frozen workload, pinned CUDA environment, current all-in pricing, independent shutdown and actual delivery of bucket-scoped worker credentials. This phase used no credentials. Evidence is sealed locally with registerv23 and an additive retention manifest; configuration draft saving does not itself publish or prove future restoration.
