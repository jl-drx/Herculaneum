# PHerc0139 public geometry proposal: technical review

Technical **GO for the proposed dependency, byte-range and reconstruction contract**, bound to query-plan SHA256 `4f180f4507e77c8460c5b3a57c0683e818c8b29957c1a6d75c70dc9d53f6623d`. There are no technical proposal blockers. This is not a live-execution GO: the proposal remains `ready=false`; its production implementation, runtime arithmetic, transport fixtures and resource enforcement need a separate frozen source review and root authorization.

The review read retained source, JSON/structural metadata and synthetic evidence only. It acquired no data and read no real XYZ coordinate payload, CT, labels, checkpoint or model array. New files are confined to this review directory. Root retains the scientific scope decision.

## Verified mapping and byte extent

The source mapping was checked against the retained exact renderer commit `650bd56663efb7225fdba73fe267ba86b8e29368`: renderer lines257–266 and827–835; loader lines82–96,365–368,222–260,263–313 and752–775. The full-canvas half-pixel origin, FP32 center, tile-local origin, double scratch-offset calculation followed by FP32 conversion, and cropped four-pixel scratch padding agree with the proposal for unit scale/crop0 and the declared128-square tiles.

The independent checker uses Python `struct` rounding rather than the author's NumPy index implementation. It enumerated all1,536 positions in each of four axes under separate multiply/add and one-final-rounding multiply/add. Fraction arithmetic verifies that its double intermediate exactly represents the unrounded expression for every simulated fused case. All declared extrema and indices match. The two forms choose the same bilinear and nearest indices throughout; they do not establish equality under arbitrary compiler transformations.

The resulting dependency rectangles are rank01 `[357,972,80,80]` and rank02 `[716,1023,80,80]`. For every XYZ TIFF the exact byte ranges are respectively5615708–6072331 and5911720–6368343 inclusive. All source identity/layout pins match the retained structural contracts: little-endian FP32,1444×1404, one uncompressed strip starting8, payload8,109,504 bytes, object8,109,638 bytes. All proposed bytes lie in that strip.

Six456,624-byte responses total2,739,744 bytes. Reserving one additional byte per request gives2,739,750, below the4MiB global ceiling. Only153,600 bytes encode selected coordinates;2,586,144 bytes remain opaque gaps. Decoding80 separate320-byte slices at stride5,776 reaches exactly the response end. A synthetic wrapper rejects every byte-read slice outside those rows; its fixture passes. Overlapping broad responses grant no additional coordinate support.

## Stencils, normals and depths

Loader1855–1874 invalidates z≤0; renderer1456–1461 invalidates x=-1. The reconstruction explicitly adds finite checks and requires every bilinear vertex, nearest center and axial normal neighbor to be valid. This is a conservative public-mesh contract; it is deliberately stricter than some cached renderer acceptance paths. Loader850–893 plus Geometry28–59 establish nearest cached normals from four axial neighbors, an interior grid requirement, cross product and normalization. Renderer278–287 establishes a single flip on the private direction followed by second normalization. The proposal correctly distinguishes all these dependencies.

Synthetic planar geometry verifies the bilinear position and normal orientation; tests reject invalid bilinear/nearest/normal-only neighbors, NaN/Inf, z≤0, x=-1, border normals and degeneracy. Each of the16 fine contributors independently affects the coarse all-valid reduction. The retained preserve-Z two-by-two reduction source supports including all16 fine samples for each level2 output; this review makes no equality claim between two rounded intensity reductions and a single mean.

Source planes21..88 give offsets−33..34 around the109-plane center54. For finite fixed b,d, staged `RN32(b + RN32(d*o))` is monotone in o in the direction of d. Therefore its finite endpoint extrema bound all68 computed depths, including zero/signed-zero directions. With native dimensions below2^24, the strict condition0≤v<dimension−1 guarantees all eight integer corners. An independent4,116-case fixture compares all68 depths with endpoint decisions, including exact/adjacent native boundaries, signed zero and subnormals; all agree. This supports the declared staged arithmetic, not authenticated historical binary equivalence.

The original catalog frame remains XYZ `[26511,26511,76953]`, volume20260102150214. No successor substitution is permitted. Mesh support and mesh-plus-declared-native-bounds are separate outputs. These tests cannot establish observed native coverage, material/tissue support, historical private masks, ink or transcription.

## Execution boundary

The plan specifies six sequential requests, no new headers/retries/redirects/multipart/full-object fallback, exact strong If-Match/206/range/length/total/identity checks and durable reservation before opening. The proposed limits are30seconds per request,240seconds external wall,512MiB RSS and32MiB output, with sequential128×128 fine tiles. Their numerical budgets are consistent, but this planning review does not certify an as-yet-unreviewed implementation enforces them.

Before a live run, the separately reviewed driver must pin these exact dependencies and arithmetic, reject all indexing outside the windows, persist failed reservations/partial outcomes, test malformed transport and gap handling, and bind the root execution GO to its final source and plan. No existing allocation-study mask/objective/decision changes follow from this review.

Evidence: `independent_checks.py` and `independent-fixtures.json` report113 passing checks. Author fixtures were inspected and hash-verified without executing their output-writing script. The seal records all local review artifacts and exact proposal identities.
