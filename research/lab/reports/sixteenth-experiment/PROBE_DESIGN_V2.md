# Additive TIFF byte-order audit after the retained unsupported-format result

This second frozen probe design supersedes only the original little-endian format restriction and initial-header schedule. Keep PROBE_DESIGN.md (SHA 95d7879c4efafb6b8cc54e6887cfef0e9ad9d4800e05ff90a38ed74a74c8466c), its exact implementation/review/plan and failed run unchanged. All source URLs, fragment/crop roles, orientation/sample/strip requirements, eight-row serving test, transport identity checks, accounting rules, downstream prerequisites and spending restrictions otherwise remain in force.

## Observed diagnosis, before any pixel reads

The first planned request returned HTTP206 bytes 0–7/103571720 with strong quoted ETag "64118120-62c6108". The eight retained bytes are 4d4d002a00000008: classic big-endian TIFF (MM, version 42), first IFD offset 8. The original classic-little-endian-only reader correctly stopped. Its run has one successful transport attempt, 8 retained/charged response bytes, zero IFD reads, zero CT pixel reads/decodes and zero model calls. No fallback, comparator or serving probe was executed.

Pin the original header-run/result.json and request-ledger.json, and raw/frag1-header.bin SHA 39740c796de25c70b056b87da657c7fdf56b5b1eb841a25ed0f3e504ff7013d8 (8 bytes), in the new ready plan. Derive the carried Frag1 identity only from the completed, hashed original request receipt. Do not fabricate a new successful header-audit result from the failed one.

This revision follows a concrete metadata diagnosis, not an unchanged automatic retry or a response to image intensity/model results. No pixels have been observed. Adding declared byte-order handling does not change uint16 values, scientific pooling, sample selection, depth or model methodology.

## Revised metadata schedule and parser

Use a new retrieval-v2 implementation and separate source-bound technical GO. Preserve the old source directory without edits. Reuse the exact already retained eight-byte Frag1 prefix and its ETag/object length; do not refetch that prefix. For Frag1 request only the two-byte IFD count at decoded offset 8, then the exact IFD table. Both requests carry the inherited If-Match and require matching strong ETag/total. For Frag2 and Frag3, retain the original prefix/count/table sequence, establishing a separate object identity per file. Order remains Frag1, Frag2, Frag3. Intended new metadata request count is 8, maximum 11 attempts; metadata budget remains 64KiB charged bodies, 16KiB per response, 180s controller. Previous 8 bytes/one attempt remain additional retained history, not erased or reset out of totals.

Accept either classic II/version 42 or MM/version 42. Select the declared struct byte order once from the exact header, and use it consistently for the IFD offset/count, every SHORT/LONG scalar, out-of-line metadata extents and next-IFD pointer. Record the observed byte_order and NumPy sample dtype '<u2' or '>u2'. Reject mixed/invalid magic, BigTIFF, bad offsets/types/counts, unsupported layout, duplicate tags, overlapping metadata/pixels and changed validators exactly as before. Inline SHORT values must be decoded from the first two bytes of the four-byte value field in the declared order; masking a big-endian LONG interpretation is incorrect. No numeric pixel decode occurs in this probe.

Synthetic validation must include asymmetric both-endian offsets and SHORT/LONG values, big-endian nonzero inline padding, exact endian-independent dimensions/strip lengths, mixed/invalid magic and existing malformed/default/extent cases. Test reuse of the prior eight-byte header only when its original URL/range/hash/HTTP206/ETag/total are pinned, and reject mismatched carried identities. Reuse unchanged transport/multipart logic only with its exact source pins or explicitly reviewed copied source. All old fixtures/receipts remain retained; new fixture records are additive.

## Serving stage and downstream interpretation

After all three revised header audits pass, make a separate ready serving plan binding the actual V2 header result and observed layouts. The original exact eight-row multipart-versus-contiguous comparison, six-attempt/512KiB/256KiB-per-response/180s limits and zero retries still apply. Byte comparison is endian-neutral because it compares identical raw sample bytes. Do not claim that the small probe validates production part counts or all 153 layers.

Combined new V2 metadata/serving charged-body cap is 576KiB and 17 attempts. Overall retained history additionally includes the prior 8-byte/one-request format rejection. Bound V2 acquisition/plan/report output at 8MiB. No larger-part test, production CT acquisition, numeric CT decoding or model call is authorized by this amendment.

A later production input contract must take the uint16 byte order from each validated TIFF, then cast decoded samples into native uint32 before exact 27-value pooling. Do not reinterpret big-endian bytes as little-endian or claim old Frag4 TIFF layout equivalence. Fixed plane 7..57, source orientation 1, source centers, division 6939 and float32 output units remain unchanged. Scientific calibration/evaluation roles and the fixed-model research design are unchanged. GPU spend remains GBP 0 and GBP 10 TOTAL first-pilot cap persists.
