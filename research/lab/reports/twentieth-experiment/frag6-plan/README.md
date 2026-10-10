# Frag6: bounded geometry acquisition proposal

**The next useful action is a small structural check of `result.ppm` and surface TIFFs `00`, `32` and `64`.** It can establish the canvas, serialization layout and any embedded rendering metadata without acquiring an image or native volume. This directory supplies the staged proposal; it is not an execution authorization, working acquisition implementation or completed geometry contract.

Four public HEAD requests completed successfully on 8 October 2026 at 23:39 UTC. `result.ppm` is **2,636,777,593 bytes**; each checked TIFF is **109,865,915 bytes**. All four responses advertise byte ranges and have strong quoted ETags. These are HTTP object validators, not content hashes or proof of immutable historical data. No response body was read, and advertised range support has not yet been tested. The complete headers, URLs and times are in [the ledger](head-request-ledger.json).

The proposed acquisition is deliberately staged:

| Stage | Concrete operation | Maximum requests | Reserved response bytes |
| --- | --- | ---: | ---: |
| A | PPM 1,024-byte prefix; three classic TIFF headers and IFDs; at most one bounded ImageDescription per TIFF | 13 | 202,301 |
| B | Fixed 9 × 9 canvas lattice, with one 3 × 3 PPM neighborhood per site | 1 | 222,640 |
| C | Conditional declared-native reconstruction, including a native slice inventory, sparse TIFF metadata and pixels, and a sign-reversed diagnostic | 436 | 4,663,451 |
| **Future total ceiling** | All three stages, only if their respective gates are satisfied | **450** | **5,088,392** |

These bounds include conservative failed-request reservations and MIME overhead. They describe response bodies, not total network traffic. Reads are sequential and streamed; retained artifacts are capped at 32 MiB, measured RSS at 256 MiB, process address space at 1 GiB, and aggregate externally controlled wall time at 30 minutes. All stages forbid retries, redirects and full-file fallback. A changed ETag, unsupported layout, incomplete body or exceeded limit stops the acquisition and preserves the failure. The four already completed HEAD requests are separate and read zero body bytes.

Stage A has exact initial URLs/ranges and deterministic, bounded IFD-following rules in [the JSON proposal](GEOMETRY_ACQUISITION_PLAN.json). Its implementation still needs independent synthetic transport/parser fixtures and a root-reviewed source freeze. Pixel bytes that happen to fall inside the PPM prefix or structural buffers remain uninterpreted. An absent TIFF description is a valid observation; it does not permit borrowing another fragment's recipe.

Stage B uses positions derived solely from canvas dimensions. Each axis is `floor((i+1)*(length−1)/10)`, for `i=0…8`. No labels, model scores or attractive-looking regions determine sampling. The PPM reader must authenticate byte order or report a uniquely invariant-consistent interpretation under the explicitly qualified phase 17 rules. All-six-zero records remain unmapped; invalid points are recorded, not replaced. This does not turn plausible coordinates into historical provenance.

**Stage C is currently blocked by missing source/render evidence.** It requires an independently declared native UUID/frame, raster map, offsets for the three sampled planes, normal handling, interpolation and quantization. The proposed native comparison uses nine predetermined lattice sites. It tests the declared recipe and its exact offset-sign reversal; it does not search for a recipe that improves agreement. Numerical tolerances must come from the declared implementation and independent synthetic error bounds before actual native pixels are read.

Matching the declared primary scan would establish sparse compatibility, not authenticate the historical rendering execution or exclude the other three energy/resolution scans. Three of those native volumes have identical declared shapes. Their bounds cannot identify the scan. The published 53-to-70 keV affine is neither an optical alignment nor a license to silently change sources. Signed order remains unresolved if the native values do not distinguish the two declared-direction tests. Even a clear result describes displacement relative to the stored normal; it does not establish inward/outward anatomy or microscopic ink depth.

Only three planes would receive numerical checks. The other 62 plane headers and pixels remain unverified; this proposal does not finish a 65-plane inference-input contract. The count of 65 files does not establish zero-offset plane 32, unit steps, a Paris `[7,58)` window, or factor-three pooling.

Frag6 remains reserved at the PHerc51/PHerc0051 parent level, across aliases and all scans. No labels, masks, optical images, CT pixels, numerical PPM records, models or paid resources were used in this planning work. No new terms were accepted and nothing was uploaded. If later geometry succeeds while annotation and upstream exposure remain unresolved, the permitted claim remains a prospectively locked comparison against supplied labels with those qualifications—not independent confirmation or authenticated text. The total first GPU pilot cap remains **£10**, with **£0** spent here.

Source hashes are retained in [SOURCE_PINS.json](SOURCE_PINS.json). Earlier reports and tracked repository files were not modified.
