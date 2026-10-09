Phase17 bounded IR and sparse PPM acquisition
===========================================

Only this additive directory contains acquisition changes. Earlier reports and repository source remain inputs. The helpers neither decode image/PPM numerical payloads nor call models.

Stages
------

1. `headers-ir`: exact9 GET requests, in retained fragment order. Each fragment contributes its listed `ir.png` as HTTP200, and exact bytes0–1023 of both listed `result.ppm` and `surface_volume.ppm` as HTTP206. Each IR body must have a strong ETag and a positive exact Content-Length ≤48MiB; aggregate IR bodies ≤96MiB. Each prefix body is exactly1024 bytes, with a1025-byte conservative failure reservation. No full PPM, volume TIFF or raw-scan request is supported.
2. `samples`: derived only after successful stage1 and pinned authoritative format evidence. Exact12 multipart HTTP206 requests, one per six frozen context cases × two distinct PPM variants. Each request samples the frozen9×9 lattice `[32,72,112,152,192,232,272,312,351]`, mapping each axis to `origin + 1 + 3*i`; each point contributes three row spans, each holding three contiguous48-byte PPM records. Exactly243 parts per request. Useful bytes34992/request and419904 total; maximum reservation222640/request and2671680 total. Multipart framing allowance is768 bytes/part +1024 bytes/request. No lattice adaptation, alternative transform, follow-up range, retry or fallback is available.

Each stage has its own source-pinned plan and fresh execution directory; the runner never advances stages automatically. Root must finish and freeze the design and complete authoritative PPM format evidence before freezing executable plans. The sample plan records all81 native centers and all243 exact spans for each case/variant. IR and sparse PPM data must remain scientifically unopened until root's separate analysis gate permits decoding.

Root CLI
--------

From this directory, after all acquisition source and fixtures are final:

```sh
python prepare_acquisition.py --stage headers-ir --design ../RESEARCH_DESIGN.md --output-dir header-plan
python run_acquisition.py --plan header-plan/acquisition-plan.json --output-dir header-run --execute
python prepare_acquisition.py --stage samples --design ../RESEARCH_DESIGN.md --headers-result header-run/result.json --format-evidence ../source-review/PPM_FORMAT_EVIDENCE.json --output-dir sample-plan
python run_acquisition.py --plan sample-plan/acquisition-plan.json --output-dir sample-run --execute
```

The design is pinned to SHA256 `08bc48f203d6722216c70d54b961d0d29ce59204359da49d2b6110661a9a9cc1`. File records consistently contain absolute `path`, exact `bytes`, and `sha256`; prior `size_bytes` records are accepted on verification and normalized. Plan code/input hashes and exact request schedules are checked before execution. Every request first writes a durable conservative reservation; successful bodies charge actual retained bytes, failed attempts retain full reservation. Partial transfer bytes are retained and hashed separately. These are bounded HTTP response-body charges, not a claim about total wire traffic. Verified system TLS, no redirects, no retries, identity content encoding and exact Content-Length are required. PPM samples require `If-Match` of the prefix response's strong ETag plus consistent object total and all exact MIME part extents. Duplicate singleton response headers, duplicate/coalesced/missing/unrequested MIME parts, oversized framing and truncated transfers fail closed.

The multipart parser was adapted additively from phase16 `production_common.py` SHA256 `2b18a2e229bc773d580daadf98d9ce2870cfadc6e6e4fec804a2b3042f1ba74a`, changing allowed part count to243. The original remains unchanged. Synthetic fixtures inject in-memory response objects and do not make network calls.
