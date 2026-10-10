# Saved Stage A runtime receipt check

PASS. The saved plan, root live GO, reviewed sources, worker/controller results, request ledger and all ten retained range-body hashes are consistent. Every request used the pinned object validator and exact range; every response receipt records the expected HTTP 206, object total and content length. Request order and derived IFD spans match the fixed Stage A traversal.

The run made 10 requests, reserved 1,514 response-body bytes and retained 1,504 bytes. The controller reports 6.325 seconds and maximum observed worker RSS of 18,505,728 bytes. Those values and retained artifacts remain within the fixed caps. No retry, redirect or stage advance is recorded.

The PPM header declares a 6,205 × 8,853 canvas, 73-byte header and six-double records; its exact total extent matches. Each of the three inspected TIFFs declares the same canvas, little-endian classic TIFF, one uncompressed uint16 strip at offset 8 with 109,865,730 bytes, followed by a 12-entry IFD. Their ImageDescription tags are absent. Each IFD has a 27-byte external Software tag, which Stage A correctly did not acquire. The other 62 TIFFs were not checked.

Rendering provenance remains unresolved. TIFF byte order does not establish numerical PPM byte order. Matching storage dimensions do not establish native source volume, physical registration, offsets, normal direction, interpolation or ink depth. No image pixels or numerical PPM records were decoded during this receipt review, and no new request, model call or scientific-array read occurred.

This check validates saved receipt and hash consistency. It does not independently replay network execution or resource-limit enforcement. Exact evidence pins and checks are retained in `verification.json`.
