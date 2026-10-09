# Fixed 256-row serving probe

This separate package implements LARGE_RANGE_DESIGN.md. It consumes the exact completed V2 header and eight-row serving PASS receipts. It retains their per-object TIFF layouts, strong ETags and source lengths. Frag1 is big-endian with strip offset260; Frag2/Frag3 are little-endian with strip offset8. Addressing uses each observed layout. No numeric pixel decoder is included.

The six new requests are, in Frag1/Frag2/Frag3 order, one256-part multipart GET followed by one contiguous full-width comparator GET. The native rows are1152..1407 of each frozen first1152-square context. A comparator is issued only after exact multipart acceptance. All768 crop row payloads must equal comparator slices byte-for-byte. No new header requests, retries, redirects, alternative ranges, coalesced parts or HTTP200 fallback are allowed.

The fixed cap is6 attempts/16MiB charged bodies/6MiB per response/240seconds/2GiB address space/32MiB output. Derived worst-case reservations total13,157,891 bytes, including768*256+1024 framing per multipart and one unknown-length completion byte per comparator. Failed attempts retain their full bounded reservation; retained partial bytes are separately counted. These response-body receipts are not complete wire traffic.

Author fixtures and plan preparation use source/JSON and synthetic bytes only. Preparation also verifies hashes of retained V2 response files without interpreting their numeric content. Root alone executes the real probe, after source-bound technical GO.

Use the managed Python with bytecode disabled to preserve earlier frozen directories. Each output directory must be new:

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python /workspace/herculaneum-lab/reports/sixteenth-experiment/retrieval-large/prepare_plan.py --output-dir /workspace/herculaneum-lab/reports/sixteenth-experiment/retrieval-large/final-plan --ready --review /workspace/herculaneum-lab/reports/sixteenth-experiment/review/retrieval-large/code-go.json --review /workspace/herculaneum-lab/reports/sixteenth-experiment/retrieval-large/fixtures-v1/fixture-receipt.json
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python /workspace/herculaneum-lab/reports/sixteenth-experiment/retrieval-large/run_large_probe.py --plan /workspace/herculaneum-lab/reports/sixteenth-experiment/retrieval-large/final-plan/probe-plan.json --output-dir /workspace/herculaneum-lab/reports/sixteenth-experiment/retrieval-large/run
```

The ready builder requires a GO binding all five exact executable files. Additional root runtime/review JSON records can be supplied with repeated `--review`. It pins their nested file records along with the scientific design, V2 receipts, prior plans/inputs, raw metadata and eight-row bodies. The runner verifies pinned inputs before and after requests. Failure is retained in `run/result.json` and `request-ledger.json`; there is no automatic retry.

A PASS applies only to these three plane07 objects and this256-part request size. Production requires a separate reviewed metadata-first acquisition plan for all remaining150 planes, with separate request/storage/concurrency budgets. This probe neither decodes CT data nor runs a model. It preserves all parent roles, source restrictions and no-publication rules; no paid compute or uploads occur.
