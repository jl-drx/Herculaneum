# Additive retrieval V2 technical GO

GO binds only the five executable files listed in code-go.json and frozen PROBE_DESIGN_V2 b19be5cc…. The original LE-only source, review, plan and stopped run remain unchanged. V2 expands declared classic TIFF byte order and reuses the already retained Frag1 header; all other strict layout/transport/multipart rules remain bounded as before.

The parser chooses II/MM from classic magic42, applies that order to offset/count/type/count/LONG/next-pointer, and decodes inline SHORT from the first two field bytes. Observed byte_order and sample dtype are explicit. BigTIFF and mixed magic fail. Asymmetric both-endian fixtures with nonzero trailing SHORT padding pass; no numeric CT sample decoding is used.

Prior-header reuse pins the original STOP result, exact single successful206 receipt, URL/range0–7, content length8, strongETag/object total and header SHA. The original raw prefix is copied locally, not refetched. Frag1 requests only IFD count/table with inherited If-Match. Frag2/3 retain prefix/count/table. New intended metadata requests8/max11;64KiB/16KiB/180s limits remain. Prior1request/8bytes stay separate history. The serving stage requires its own new ready plan after all V2 headers pass; eight-row scope and caps remain unchanged.

Copied range_http.py is byte-identical to the reviewed original00f23401… . Common multipart parsing is unchanged. All original fail-closed TIFF extent/default, range identity, partial retention, failure charge, no200body fallback, binary-length MIME and stage-barrier checks remain in force.

Independent50-case synthetic suite PASS, including49 parser/transport cases and a mocked carried-header execution proving exactlytwo new IFD requests with no prefix GET. Author57 cases independently rerun PASS, including mismatched carriedstatus/range/URL/hash/ETag/total and both-endian scalar/extent cases. No real network, pixel samples, arrays, checkpoints or models were accessed; review read existing JSON metadata receipts only.

This is source conformance, not a successful real V2 audit or production/CT/inference readiness claim. Preserve any new unsupported/failure result; no automatic retry or fallback. Root owns all actual calls and separately frozen plans.
