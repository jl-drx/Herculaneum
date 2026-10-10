# Independent current-public geometry runtime verification

This contract is fixed before reading any acquired XYZ coordinate values. It covers only the two PHerc0139 contexts in proposal SHA256 `4f180f4507e77c8460c5b3a57c0683e818c8b29957c1a6d75c70dc9d53f6623d`. Production source identity and reviewer source identities are bound in `CHECKER_FREEZE.json` before execution. Root must separately supply the exact successful producer result and acquisition-GO hashes. A review failure is retained without changing its numerical acceptance criteria.

No reviewer network, native CT, label, S/V, model, weight or prediction access is permitted. The reviewer reads source and structural metadata, exact six retained raw range objects and only the producer outputs named by the successful pinned result. Hashing raw bytes does not interpret them as coordinates.

## Transport and selected bytes

Independently derive every inclusive range from the authenticated uncompressed little-endian float32 single strip: byte offset 8, width 1444, height 1404, four bytes per sample. Require the six fixed ordered identities, source URLs, unchanged strong ETags, exact HTTP 206 Content-Range/total, identity encoding, declared length when present, byte count and body SHA256. Require 2,739,744 retained body bytes and 2,739,750 reserved bytes across six requests, with no additional attempts in the retained ledger. Pins and receipts substantiate retained execution evidence; the reviewer did not observe the original HTTP sessions or filesystem crash durability directly.

Decode exactly 80 selected row slices of 320 bytes from each body, at relative stride 5776. Decode each selected word through an independent struct integer-bit reader and a float32 view so NaN payloads and signed zero are preserved. Compare both saved 80×80×3 windows bit for bit. Only 153,600 coordinate bytes are interpreted. The 2,586,144 gap bytes remain opaque. Do not decode overlapping broad-span gaps as new geometry.

## Independent numerical reconstruction

The reviewer imports no producer computational functions. A struct-rounded scalar axis implementation independently derives all 1,536 positions per axis, bilinear indices and nearest normal centers, including the alternate one-FMA dependency envelope. Require the exact 80×80 dependency windows without extending them.

A separate NumPy implementation processes 64-row bands across the complete 1,536×1,536 fine contexts. It checks finite XYZ, x=-1 and z≤0 sentinels; all four bilinear vertices; the nearest five-vertex axial normal stencil; staged float32 cross-product norm, inverse normalization, normal flip and second normalization; finite base/direction; and strict complete trilinear-corner bounds in the declared original native frame `[26511,26511,76953]`. Bounds use offsets −33 and +34 under the predeclared monotonicity proof covering all 68 integer depths. Logical horizontal then vertical four-way reductions require all 16 fine constituents per coarse pixel.

Independently reconstruct **all 4,718,592 fine samples**, four complete 384×384 output masks, and each context's complete disjoint fine-rejection counts. All boolean masks, integer counts, subset relations and partition sums must agree exactly. No mask mismatch allowance or floating count tolerance is permitted. Validate all output pins and dtype/shape contracts.

A separately authored struct-FP32 scalar oracle checks the Cartesian product of local coordinates `0,1,3,4,127,128,255,512,767,1024,1407,1534,1535`: 169 predetermined fine points per context, 338 in total. Require bitwise equality of finite scalar/vector bases, and directions where mesh support is accepted; require exact position, mesh and bounded-support decisions everywhere sampled. At these same points enumerate all 68 depths and require the same bound decision as the two endpoints. Record hashes of sampled intermediate tensors without exposing full additional coordinate arrays. Nonfinite rejected bases are checked through the acceptance decisions, not NaN payload equality after arithmetic.

This scalar subset is additional coverage of the independently reconstructed complete masks, not a claim of all-point scalar execution or all-depth enumeration everywhere. Endpoint monotonicity supplies the remaining depth coverage. The contract does not establish historical compiler/FMA behavior beyond the declared staged calculation.

## Acceptance, preparation and resources

The arithmetic acceptance is **exact**, with no absolute/relative float tolerance: only integer-bit/window and supported-intermediate identity, exact support decisions, counts, pins and bookkeeping are compared. Finite scalar float32 roundings are tested on generated flat, tilted and reversed planes before any actual coordinates. Adversarial generated coordinates cover invalid sentinels, nonfinite and degenerate normals, off-cell stencil failure, escaped-window rejection, strict/adjacent float bounds, and opaque gaps. A separate scalar fixture enumerates 2,048 deterministic synthetic endpoint-versus-all-depth cases. Retained transport mutations exercise changed HTTP status, ETag, range, encoding, length, reservation, hash and URL.

Pin Python 3.12.14 and NumPy 2.5.3, use one numerical thread, and read no native CT samples. Verify producer reported RSS/wall checks and actual retained-directory bytes against its frozen limits; this is not an independent measurement of the producer's transient peak RSS or proof of OS-enforced limits. Retain reviewer timing and peak RSS separately. Source and input pins are checked before and after reconstruction. No scientific retry, support expansion, inferred background, mask application to the completed allocation study, or tolerance revision is authorized.

## Interpretation boundary

A PASS establishes consistency of the retained public-coordinate reconstruction under the declared conservative staged arithmetic. It does not authenticate historical private rendering, historical mask presence, native voxel availability or tissue/material support, annotation or teacher ancestry, ink, text, translations, model efficacy or new training support. Root owns scientific interpretation and any separate future authorization.
