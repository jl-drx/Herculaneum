# Bounded retrieval probe

This package probes plane 07 on the three source TIFFs fixed by the sixteenth design. It does not assemble scientific arrays, decode uint16 samples, call models, acquire the 51-plane inputs or choose a production range count. Root executes both stages after technical review; preparation used only source metadata and synthetic binary fixtures.

Stage A follows the predeclared adaptive rule: 8-byte classic-TIFF prefix, 2-byte IFD count at its pointer, then exactly `2+12*N+4` IFD bytes for `1<=N<=128`. The intended total is nine GETs, capped at 12 attempts, 64 KiB charged bodies, 16 KiB per response and 180 seconds. It rejects unsupported layout without introducing another decoder. Required scalar tags and specification defaults are explicit; out-of-line auxiliary metadata extents are validated without dereferencing them. No prefix pixel bytes are intentionally requested.

Stage B requires a separate plan binding all three successful Stage A object contracts and raw metadata receipts. It issues eight crop-width row ranges in one multipart GET, followed by one eight-full-row comparator GET, for each fragment in order. Rows are the first eight rows of each fixed first context; the comparator is never reached after a failed multipart response. Raw bytes must match exactly. Multipart parts retain original body offsets, sizes, ranges and hashes without numeric pixel interpretation.

HTTP206, exact bounds/length, identity encoding and a quoted strong ETag are required. After the first Stage A range, every request sends the same If-Match; Stage B sends the audited value from its first request. HTTP200 is rejected before reading its body. Redirects, retries and full-object fallback are disabled. Singleton framing/identity headers cannot be repeated. A single valid leading CRLF is permitted in multipart framing; arbitrary preambles are rejected. Binary payloads are sliced by exact requested length, including any delimiter-like bytes they contain.

The multipart payload cap is 18,432+8,192 framing bytes per fragment. Comparator caps are exact eight-full-row lengths plus one byte for bounded EOF/overlength detection when Content-Length is absent. Six reserved caps sum to 417,235 bytes, below the 512 KiB Stage B limit. Successful requests charge actual complete bodies; failed attempts retain their full reserved charge. Partial stored bytes are recorded separately, including IncompleteRead.partial. These are bounded response-body accounts, not total wire traffic. Request receipts include elapsed time and fixed nonsecret validation stop reasons.

Root commands from this directory, using fresh output directories:

```bash
/workspace/.venvs/herculaneum/bin/python prepare_probe_plan.py \
  --stage headers --output-dir headers-plan --ready \
  --review ../review/code-go.json --review code-freeze.json \
  --review fixtures-v1/fixture-receipt.json
/usr/bin/timeout 180 /workspace/.venvs/herculaneum/bin/python run_probe.py \
  --plan headers-plan/probe-plan.json --output-dir headers-run
```

Root inspects `headers-run/result.json`. Only after all three objects pass, create and inspect the separate Stage B plan before its requests:

```bash
/workspace/.venvs/herculaneum/bin/python prepare_probe_plan.py \
  --stage serving --header-result headers-run/result.json \
  --output-dir serving-plan --ready \
  --review ../review/code-go.json --review code-freeze.json \
  --review fixtures-v1/fixture-receipt.json
/usr/bin/timeout 180 /workspace/.venvs/herculaneum/bin/python run_probe.py \
  --plan serving-plan/probe-plan.json --output-dir serving-run
```

Add any root runtime/review JSON receipts as additional `--review` arguments. A ready plan requires a top-level GO receipt binding all five exact source files. The draft header plan remains `ready=false`; no serving plan can exist before observed header evidence. Raw bodies, request-ledger.json and result.json remain in each stage directory on success or failure. Stop on failure, preserve evidence and do not retry, coalesce parts, alter the crop or substitute a source.

Maximum combined raw acquisition is 576 KiB across both stage caps and 18 attempts. Probe output overhead is bounded at 8 MiB. The observed eight-part result, even if successful, establishes neither larger multipart support nor production/scientific input readiness. Any larger-count or 153-plane acquisition needs its own frozen plan. Roles, source-license restrictions, unmeasured optical registration and unknown upstream exposure remain unchanged.
