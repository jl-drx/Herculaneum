# Additive classic TIFF byte-order probe

The original retrieval source, review, plan and stopped run remain unchanged. Its valid HTTP206 eight-byte Frag1 prefix was 4d4d002a00000008: classic big-endian TIFF with first IFD offset8. It made no IFD, pixel, serving or model request. That original receipt remains STOP; V2 creates separately reviewed evidence under PROBE_DESIGN_V2.md.

V2 accepts classic II or MM/version42, using the declared byte order for offsets, counts, fields and next-IFD pointers. Inline SHORT values use the first two field bytes, including nonzero padding. Observed contracts record byte_order and '<u2' or '>u2' for a future decoder. No numeric samples are decoded here. BigTIFF, unsupported layouts/types, ambiguous tags, invalid extents and changed validators still stop.

Stage A verifies and reuses the exact Frag1 prefix and original HTTP/hash lineage; it never refetches that prefix. Its next two IFD requests carry inherited If-Match. Frag2/3 retain prefix/count/table sequences. Intended new GET count is8, maximum11; metadata budget remains64KiB charged bodies,16KiB per response and180 seconds. Previous history remains separately visible as one request/eight bytes. Source preparation and57 author synthetic cases used no real network, CT pixel reads, numeric arrays or model calls. The transport source is byte-identical to the original; independent50-case review includes a no-prefix-GET cache simulation.

Root commands from this directory, with fresh output directories and the source-bound V2 GO:

```bash
/workspace/.venvs/herculaneum/bin/python prepare_probe_plan.py --stage headers --output-dir header-plan --ready --review ../review/retrieval-v2/code-go.json --review code-freeze.json --review fixtures-v1/fixture-receipt.json
/usr/bin/timeout 180 /workspace/.venvs/herculaneum/bin/python run_probe.py --plan header-plan/probe-plan.json --output-dir header-run
```

Add root runtime/review JSONs with additional --review arguments. The ready plan pins V2 design, five executed files, source/benchmark contracts and the exact original STOP result/ledger/prefix. The draft header plan remains ready=false. Only after all three V2 header audits pass, root creates and inspects a separate serving plan before any of its requests:

```bash
/workspace/.venvs/herculaneum/bin/python prepare_probe_plan.py --stage serving --header-result header-run/result.json --output-dir serving-plan --ready --review ../review/retrieval-v2/code-go.json --review code-freeze.json --review fixtures-v1/fixture-receipt.json
/usr/bin/timeout 180 /workspace/.venvs/herculaneum/bin/python run_probe.py --plan serving-plan/probe-plan.json --output-dir serving-run
```

Stage B remains exactly eight crop rows in one multipart GET followed by eight full rows in one comparator GET per fragment: six attempts,512KiB combined,256KiB per response,180 seconds. Derived six-response caps total417,235 bytes. Comparison is endian-neutral raw-byte equality. No comparator follows a failed multipart response. HTTP200 is rejected before body reads; validators/If-Match, exact lengths, partial retention, conservative failure charges, nonsecret validation reasons and timings remain enforced. There are no retries, redirects, full-object fallback, numeric decoding or model calls.

V2 stage caps total576KiB and17 attempts, with the original one-request/eight-byte history additional and visible; overhead is bounded at8MiB. A passing eight-row probe does not establish larger-part support, all153-plane readiness, scientific arrays or accuracy. Any larger probe/production acquisition requires its own frozen plan. Roles, EduceLab license/citations and unknown upstream exposure remain unchanged.
